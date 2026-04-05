import os
# Must be before import torch to block fragmentation native to the caching allocator
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Removed for Windows stability

import torch
import gc
import json
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer, get_linear_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image
import bitsandbytes as bnb

# Import the architecture
from model import ClinicalMicroVLM

class ClinicalChartDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor, tokenizer, max_samples=100000):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.tokenizer = tokenizer
        
        print(f"Loading dataset from {image_dir}...")
        
        # Dictionary to store samples by category for balanced sampling
        category_samples = {}
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            raise FileNotFoundError(f"Ensure that {image_dir} and {label_dir} exist.")
            
        # 1. Collect all valid samples by category
        for root, dirs, files in os.walk(label_dir):
            category = os.path.basename(root)
            if category == os.path.basename(label_dir):
                continue # Skip the root labels folder
                
            if category not in category_samples:
                category_samples[category] = []
                
            for label_file in files:
                if not label_file.endswith('.json'):
                    continue
                base_name = os.path.splitext(label_file)[0]
                
                # Check for image in corresponding category folder
                img_path = os.path.join(image_dir, category, f"{base_name}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(image_dir, category, f"{base_name}.jpg")
                    if not os.path.exists(img_path):
                        continue
                
                category_samples[category].append((img_path, os.path.join(root, label_file)))

        # 2. Perform Equal Balanced Sampling across ALL categories
        num_categories = len(category_samples)
        if num_categories == 0:
            raise ValueError(f"No valid label categories found in {label_dir}")
            
        samples_per_category = max_samples // num_categories
        
        final_samples = []
        
        for cat, samples in category_samples.items():
            if len(samples) >= samples_per_category:
                final_samples.extend(random.sample(samples, samples_per_category))
            else:
                print(f"WARNING: Not enough samples for '{cat}' ({len(samples)}). Taking all.")
                final_samples.extend(samples)
                
        # 3. Final Shuffle
        random.shuffle(final_samples)
        self.samples = final_samples[:max_samples] # Ensure exact max_samples count
        
        print(f"Dataset initialized with {len(self.samples)} samples (Equal Balance across {num_categories} categories).")
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        # -- 1. Load and process Image with Fault Tolerance --
        try:
            image = Image.open(img_path)
            image.verify() # Verify file integrity
            
            # verify() alters internal state, reopen it to convert
            image = Image.open(img_path).convert("RGB")
            
            # 1. Global Context (Resize whole image)
            global_img = image.resize((384, 384))
            
            # 2. Local Context (Resize to 768x768, then chop into 4 quadrants)
            img_768 = image.resize((768, 768))
            tl = img_768.crop((0, 0, 384, 384))
            tr = img_768.crop((384, 0, 768, 384))
            bl = img_768.crop((0, 384, 384, 768))
            br = img_768.crop((384, 384, 768, 768))
            
            # Pass all 5 to processor
            images = [global_img, tl, tr, bl, br]
            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
            
            # -- 2. Load Label --
            with open(label_path, 'r', encoding='utf-8', errors='replace') as f:
                target_json = f.read() # Target label is the raw string content
                
        except Exception as e:
            print(f"\nWARNING: Skipping corrupted image {img_path}: {e}")
            with open("corrupted_images.log", "a") as err_log:
                err_log.write(f"{img_path}\n")
            # Recursively try next image
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)

        # -- 3. Dynamic Router Seeding --
        if random.random() < 0.05: # 5% of the time, teach it to classify
            user_prompt = "\nClassify this chart type. Output only the exact schema name.\n"
            try:
                parsed_json = json.loads(target_json)
                target_json = str(parsed_json.get("chart_type", "unknown"))
            except json.JSONDecodeError:
                target_json = "unknown"
        else: # 95% of the time, do standard extraction
            user_prompt = "\nExtract the underlying data from this clinical chart in strict JSON format.\n"
            
        # -- 4. Formulate Prompt and Tokenize --
        full_text = user_prompt + target_json + self.tokenizer.eos_token
        
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=768, # Reduced from 1024 (Stage 3) to prevent Windows VRAM thrashing
            return_tensors="pt"
        )
        
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        
        # -- 5. Formulate Labels (Mask out prompt) --
        labels = input_ids.clone()
        prompt_encoded = self.tokenizer(user_prompt, add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_encoded.input_ids.shape[1]
        
        labels[:prompt_len] = -100 # Mask prompt
        labels[attention_mask == 0] = -100 # Mask padding
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def main():
    # --- Configuration & Hyperparameters ---
    IMAGE_DIR = r"C:\sem4\KMVision-1 Data\dataset\images"
    LABEL_DIR = r"C:\sem4\KMVision-1 Data\dataset\labels"
    # Projector weights for Phase B initialization (updated per user feedback)
    PROJECTOR_WEIGHTS_PATH = r"C:\sem4\KMVision-1\checkpoints\checkpoints_projector\projector_weights.pth"
    CHECKPOINT_DIR = r"checkpoints/phase_b/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Pre-create directory early
    
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 16 
    LEARNING_RATE = 5e-5
    EPOCHS = 1
    SUBSET_SIZE = 100000
    
    device = torch.device("cuda:0")
    print(f"Executing STRICTLY on device: {device}")
    
    try:
        torch.empty(1, device=device)
    except Exception as e:
        print(f"FATAL CUDA ERROR: {e}")
        return
    
    print("Loading models and processors...")
    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
        
    model = ClinicalMicroVLM(bnb_config=bnb_config)
    
    print("Applying Phase B Freezing Constraints and LoRA Adapters...")
    model.vision_encoder.requires_grad_(False)
    # Apply LoRA Config (Rank 64 for VRAM survival)
    lora_config = LoraConfig(
        r=64, 
        lora_alpha=128, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        task_type="CAUSAL_LM"
    )
    from peft import get_peft_model
    model.llm = get_peft_model(model.llm, lora_config)
    
    model.llm.gradient_checkpointing_enable()
    model.llm.config.use_cache = False 
    model.projector.requires_grad_(True)
    
    print(f"Loading Projector weights from: {PROJECTOR_WEIGHTS_PATH}")
    if os.path.exists(PROJECTOR_WEIGHTS_PATH):
        model.projector.load_state_dict(torch.load(PROJECTOR_WEIGHTS_PATH, map_location=device))
        print("Successfully loaded pre-trained projector weights.")
    else:
        print(f"WARNING: Projector weights not found at {PROJECTOR_WEIGHTS_PATH}. Starting fresh.")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (LoRA + Projector): {trainable_params:,}")
    
    model.vision_encoder = model.vision_encoder.to(device)
    model.projector = model.projector.to(device)
    
    # Paged 8-bit AdamW for VRAM saving (offloads optimizer states to CPU)
    optimizer = bnb.optim.PagedAdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    dataset = ClinicalChartDataset(IMAGE_DIR, LABEL_DIR, processor, tokenizer, max_samples=SUBSET_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    total_steps = (len(dataloader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    
    print("\n--- Starting Phase B Training (QLoRA Fine-Tuning - 5-Crop Pooling) ---")
    model.train()
    torch.cuda.empty_cache()
    
    optimizer.zero_grad()
    global_step = 0
    
    try:
        for epoch in range(EPOCHS):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for step, batch in enumerate(progress_bar):
                pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Print memory diagnostics for the very first step
                if step == 0:
                    print(f"\n--- DIAGNOSTICS: Batch max sequence length: {input_ids.shape[1]} + 3645 Global/Local Image patches ---")
                    
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss_val = loss.item() * GRAD_ACCUM_STEPS
                loss.backward()
                
                del outputs, loss
                
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Clear cache strictly after gradient updates to release massive backward buffers
                    torch.cuda.empty_cache()
                    
                    # Checkpoint every 250 steps (approx every 4,000 images)
                    if global_step % 250 == 0:
                        step_dir = os.path.join(CHECKPOINT_DIR, f"step_{global_step}")
                        os.makedirs(step_dir, exist_ok=True)
                        print(f"\n[Step {global_step}] Saving intermediate weights...")
                        model.llm.save_pretrained(step_dir)
                        torch.save(model.projector.state_dict(), os.path.join(step_dir, "projector_weights.pth"))
    
                    # Check for "save_now.txt" trigger file
                    if os.path.exists("save_now.txt"):
                        print(f"\n[TRIGGER] Manual save requested via save_now.txt...")
                        trigger_dir = os.path.join(CHECKPOINT_DIR, f"manual_step_{global_step}")
                        os.makedirs(trigger_dir, exist_ok=True)
                        model.llm.save_pretrained(trigger_dir)
                        torch.save(model.projector.state_dict(), os.path.join(trigger_dir, "projector_weights.pth"))
                        os.remove("save_now.txt")
                        print(f"Manual checkpoint saved to {trigger_dir}")
                    
                if step == 0:
                    mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
                    print(f"--- DIAGNOSTICS: Max VRAM Allocated: {mem_alloc:.2f}GB / Reserved: {mem_reserved:.2f}GB ---")
                
                total_loss += loss_val
                current_reserved = torch.cuda.memory_reserved() / 1024**3
                progress_bar.set_postfix({"loss": loss_val, "vram": f"{current_reserved:.1f}G"})
                
                del batch, pixel_values, input_ids, attention_mask, labels
                
                # Stage 3: Frequency cache clearing was removed for per-accumulation clearing above.
                
    except KeyboardInterrupt:
        print("\n\n[INTRRUPTED] Training stopped by user. Saving emergency checkpoint...")
        interrupt_dir = os.path.join(CHECKPOINT_DIR, "interrupt_checkpoint")
        os.makedirs(interrupt_dir, exist_ok=True)
        model.llm.save_pretrained(interrupt_dir)
        torch.save(model.projector.state_dict(), os.path.join(interrupt_dir, "projector_weights.pth"))
        print(f"Emergency weights saved to {interrupt_dir}. Exiting safely.")
        return # Exit main() safely
                
    print("\nTraining completed. Saving Phase B weights...")
    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.llm.save_pretrained(final_dir)
    torch.save(model.projector.state_dict(), os.path.join(final_dir, "projector_weights.pth"))
    print(f"Final Phase B weights saved successfully to {final_dir}")

if __name__ == "__main__":
    main()