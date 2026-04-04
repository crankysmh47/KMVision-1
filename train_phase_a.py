import os
# Must be before import torch to block fragmentation native to the caching allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image

# Import the architecture
from model import ClinicalMicroVLM

class ClinicalChartDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor, tokenizer, max_samples=25000):
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
                
                # Constraint: Only train on the first 25,000 charts
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
            
        # -- 3. Formulate Prompt and Tokenize --
        # Note: <image> is structurally handled in model.py by prepending image embeddings.
        # Thus, our text starts after the <image> placeholder.
        user_prompt = "\nExtract the underlying data from this clinical chart in strict JSON format.\n"
        full_text = user_prompt + target_json + self.tokenizer.eos_token
        
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length", # CRITICAL TO PREVENT PyTorch Memory Leak! Do not remove!
            max_length=1536, # Boosted to safely accommodate standard complex chart JSONs
            return_tensors="pt"
        )
        
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        
        # -- 4. Formulate Labels (Mask out prompt) --
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
    CHECKPOINT_DIR = "checkpoints/phase_a_projector/"
    
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 8 # (or 16)
    LEARNING_RATE = 1e-3 # Higher LR for projector warm-up
    EPOCHS = 1
    SUBSET_SIZE = 25000
    
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
        
    # Initialize our custom architecture
    model = ClinicalMicroVLM()
    
    print("Applying Phase A Freezing Constraints...")
    model.vision_encoder.requires_grad_(False)
    model.llm.requires_grad_(False)
    model.llm.config.use_cache = False # Disable KV cache for training
    model.projector.requires_grad_(True)
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Projector only): {trainable_params:,}")
    
    # Strictly push to GPU
    model = model.to(device)
    
    # Initialize Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Prepare Dataset & DataLoader
    dataset = ClinicalChartDataset(IMAGE_DIR, LABEL_DIR, processor, tokenizer, max_samples=SUBSET_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Schedulers
    total_steps = (len(dataloader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    
    print("\n--- Starting Phase A Training (Projector Warm-Up) ---")
    model.train()
    
    # Ensure memory is clean before training loops
    torch.cuda.empty_cache()
    
    optimizer.zero_grad()
    
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
            
            # AGGRESSIVE python garbage collection of the 400MB+ logits tensor
            # If we don't delete this, it lives through the next iteration's forward pass!
            del outputs
            del loss
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
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
            
            # Periodically clear cache to combat variable-length sequence fragmentation
            if step % 20 == 0:
                torch.cuda.empty_cache()
                
    # --- Checkpointing ---
    print("\nTraining completed. Saving Projector weights...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(CHECKPOINT_DIR, "projector_weights.pth"))
    print(f"Projector weights saved successfully to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()
