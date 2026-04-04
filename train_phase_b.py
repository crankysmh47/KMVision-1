import os
# Must be before import torch to block fragmentation native to the caching allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

# Import the architecture
from model import ClinicalMicroVLM

class ClinicalChartDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor, tokenizer, max_samples=100000):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.tokenizer = tokenizer
        
        print(f"Loading dataset from {image_dir} and {label_dir}...")
        self.samples = []
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            raise FileNotFoundError(f"Ensure that {image_dir} and {label_dir} exist.")
            
        # Iterate over labels to pair with images, handling subdirectories
        for root, _, files in os.walk(label_dir):
            category = os.path.basename(root)
            for label_file in sorted(files):
                if not label_file.endswith('.json'):
                    continue
                base_name = os.path.splitext(label_file)[0]
                
                # Look for corresponding image (could be png or jpg) inside the matching category directory
                img_root = os.path.join(image_dir, category)
                img_path = os.path.join(img_root, f"{base_name}.png")
                
                if not os.path.exists(img_path):
                    img_path = os.path.join(img_root, f"{base_name}.jpg")
                    if not os.path.exists(img_path):
                        continue # Skip if no image found
                
                self.samples.append((img_path, os.path.join(root, label_file)))
                
                # Constraint: Limit samples
                if len(self.samples) >= max_samples:
                    break
            if len(self.samples) >= max_samples:
                break
                
        print(f"Dataset initialized with {len(self.samples)} samples.")
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        # -- 1. Load and process Image --
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # -- 2. Load Label --
        with open(label_path, 'r', encoding='utf-8', errors='replace') as f:
            target_json = f.read() # Target label is the raw string content
            
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
        # Note: <image> is structurally handled in model.py by prepending image embeddings.
        # Thus, our text starts after the <image> placeholder.
        full_text = user_prompt + target_json + self.tokenizer.eos_token
        
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=2048, # Fixed ceiling for memory stability
            return_tensors="pt"
        )
        
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        
        # -- 5. Formulate Labels (Mask out prompt) --
        labels = input_ids.clone()
        
        # We find the length of the prompt to mask it out from loss computation
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
    CHECKPOINT_DIR = "checkpoints/phase_b/"
    
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 16 
    LEARNING_RATE = 5e-5 # Optimized for QLoRA with gradient accumulation
    EPOCHS = 1
    SUBSET_SIZE = 100000
    
    # FORCE CUDA DEVICE
    device = torch.device("cuda:0")
    print(f"Executing STRICTLY on device: {device}")
    
    # Force initialize CUDA context to catch any silent failure logs
    try:
        torch.empty(1, device=device)
    except Exception as e:
        print(f"FATAL CUDA ERROR: {e}")
        return
    
    print("Loading models and processors...")
    # Load processor for the vision encoder
    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384", trust_remote_code=True)
    # Load tokenizer for the LLM
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True)
    
    # Configure padding token if it's missing (Qwen uses eos as pad by default in some setups)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Setting up BitsAndBytesConfig for 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
        
    # Initialize our custom architecture with 4-bit config
    model = ClinicalMicroVLM(bnb_config=bnb_config)
    
    print("Applying Phase B Freezing Constraints and LoRA Adapters...")
    # Freeze Vision Encoder entirely
    model.vision_encoder.requires_grad_(False)
    
    # Apply LoRA Config
    lora_config = LoraConfig(
        r=64, 
        lora_alpha=128, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        task_type="CAUSAL_LM"
    )
    model.llm = get_peft_model(model.llm, lora_config)
    
    # Enable Gradient Checkpointing on LLM for VRAM survival
    model.llm.gradient_checkpointing_enable()
    
    model.llm.config.use_cache = False # Disable KV cache for training
    model.projector.requires_grad_(True)
    
    # Load Phase A Projector Weights (CRITICAL)
    print("Loading Phase A Projector weights...")
    projector_path = "checkpoints/phase_a_projector/projector_weights.pth"
    if os.path.exists(projector_path):
        model.projector.load_state_dict(torch.load(projector_path, map_location=device))
        print("Successfully loaded pre-trained projector weights.")
    else:
        print(f"WARNING: Phase A Projector weights not found at {projector_path}.")
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (LoRA + Projector): {trainable_params:,}")
    
    # Strictly push to GPU (LLM is 4-bit so we only push the other components if needed)
    # the device mapping for 4-bit handles itself, but we should make sure our unwrapped components reside on CUDA
    model.vision_encoder = model.vision_encoder.to(device)
    model.projector = model.projector.to(device)
    
    # Initialize Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Prepare Dataset & DataLoader
    dataset = ClinicalChartDataset(IMAGE_DIR, LABEL_DIR, processor, tokenizer, max_samples=SUBSET_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Schedulers
    total_steps = (len(dataloader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    
    print("\n--- Starting Phase B Training (QLoRA Fine-Tuning) ---")
    model.train()
    
    # Ensure memory is clean before training loops
    torch.cuda.empty_cache()
    
    optimizer.zero_grad()
    
    global_step = 0
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            # Strictly move variables to target device and correct input types
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Print memory diagnostics for the very first step
            if step == 0:
                print(f"\n--- DIAGNOSTICS: Batch max sequence length: {input_ids.shape[1]} + Image patches ---")
                
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss_val = loss.item() * GRAD_ACCUM_STEPS
            
            loss.backward()
            
            # AGGRESSIVE python garbage collection
            del outputs
            del loss
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Intermediate Checkpointing
                if global_step % 5000 == 0:
                    step_dir = os.path.join(CHECKPOINT_DIR, f"step_{global_step}")
                    os.makedirs(step_dir, exist_ok=True)
                    print(f"\n[Step {global_step}] Saving intermediate weights to {step_dir}...")
                    model.llm.save_pretrained(step_dir)
                    torch.save(model.projector.state_dict(), os.path.join(step_dir, "projector_weights.pth"))
                
            if step == 0:
                mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
                mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
                print(f"--- DIAGNOSTICS: Max VRAM Allocated: {mem_alloc:.2f}GB / Reserved: {mem_reserved:.2f}GB ---")
            
            # Accumulate true loss
            total_loss += loss_val
            
            # Real-time VRAM tracker in postfix
            current_reserved = torch.cuda.memory_reserved() / 1024**3
            progress_bar.set_postfix({
                "loss": loss_val, 
                "vram": f"{current_reserved:.1f}G"
            })
            
            del batch, pixel_values, input_ids, attention_mask, labels
            
            # Periodically clear cache every 10 steps to maintain stability without stalling
            if step % 10 == 0:
                torch.cuda.empty_cache()
                
    # --- Final Checkpointing ---
    print("\nTraining completed. Saving Phase B weights...")
    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.llm.save_pretrained(final_dir)
    torch.save(model.projector.state_dict(), os.path.join(final_dir, "projector_weights.pth"))
    print(f"Final Phase B weights saved successfully to {final_dir}")

if __name__ == "__main__":
    main()
