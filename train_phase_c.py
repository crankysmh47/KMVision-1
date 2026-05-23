import os

import torch
import json
import random
from datetime import datetime, timezone
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer, get_linear_schedule_with_warmup, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from PIL import Image
import bitsandbytes as bnb

from model import ClinicalMicroVLM
from evaluation.schema_compact import compact_fits_token_budget, compact_json_string

EXTRACTION_PROMPT = "\nExtract the underlying data from this clinical chart in strict JSON format.\n"
CLASSIFY_PROMPT = "\nClassify this chart type. Output only the exact schema name.\n"
CORRUPT_LOG = "corrupted_images.log"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _delete_corrupt_pair(img_path: str, label_path: str, error: Exception) -> None:
    """Log and delete corrupt image/label so training never stalls on them again."""
    line = f"{_utc_now()}\t{img_path}\t{label_path}\t{error}\n"
    with open(CORRUPT_LOG, "a", encoding="utf-8") as err_log:
        err_log.write(line)
    for path in (img_path, label_path):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            print(f"WARNING: Could not delete {path}: {exc}")


def _save_checkpoint(
    model,
    checkpoint_dir: str,
    global_step: int,
    loss_val: float,
    *,
    folder_name: str | None = None,
) -> str:
    """Save LoRA adapter + projector and write a small metadata sidecar."""
    step_dir = os.path.join(checkpoint_dir, folder_name or f"step_{global_step}")
    os.makedirs(step_dir, exist_ok=True)
    model.llm.save_pretrained(step_dir)
    projector_path = os.path.join(step_dir, "projector_weights.pth")
    torch.save(model.projector.state_dict(), projector_path)

    meta = {
        "global_step": global_step,
        "loss": loss_val,
        "saved_at_utc": _utc_now(),
        "adapter_dir": step_dir,
        "projector_weights": projector_path,
    }
    meta_path = os.path.join(step_dir, "checkpoint_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    adapter_file = os.path.join(step_dir, "adapter_model.safetensors")
    if not os.path.isfile(projector_path) or not os.path.isfile(adapter_file):
        raise RuntimeError(f"Checkpoint incomplete at {step_dir}")
    return step_dir


class CompactChartDataset(Dataset):
    """Phase C: verbose labels on disk, compact JSON as training target."""

    def __init__(self, image_dir, label_dir, processor, tokenizer, max_samples=30000, seed=42):
        self.processor = processor
        self.tokenizer = tokenizer
        category_samples = {}

        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Missing {image_dir} or {label_dir}")

        for root, _, files in os.walk(label_dir):
            category = os.path.basename(root)
            if category == os.path.basename(label_dir):
                continue
            category_samples.setdefault(category, [])
            for label_file in files:
                if not label_file.endswith(".json"):
                    continue
                base = os.path.splitext(label_file)[0]
                img_path = os.path.join(image_dir, category, f"{base}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(image_dir, category, f"{base}.jpg")
                    if not os.path.exists(img_path):
                        continue
                category_samples[category].append((img_path, os.path.join(root, label_file)))

        rng = random.Random(seed)
        num_categories = len(category_samples)
        if num_categories == 0:
            raise ValueError(f"No categories in {label_dir}")

        samples_per_category = max_samples // num_categories
        final_samples = []
        for cat, samples in category_samples.items():
            if len(samples) >= samples_per_category:
                final_samples.extend(rng.sample(samples, samples_per_category))
            else:
                final_samples.extend(samples)
        rng.shuffle(final_samples)
        raw = final_samples[:max_samples]

        self.samples = []
        skipped_budget = 0
        skipped_corrupt = 0
        for img_path, label_path in raw:
            try:
                if not os.path.isfile(img_path) or not os.path.isfile(label_path):
                    skipped_corrupt += 1
                    continue
                with open(label_path, encoding="utf-8", errors="replace") as f:
                    obj = json.load(f)
                if compact_fits_token_budget(obj, tokenizer):
                    self.samples.append((img_path, label_path))
                else:
                    skipped_budget += 1
            except (json.JSONDecodeError, OSError) as exc:
                skipped_corrupt += 1
                _delete_corrupt_pair(img_path, label_path, exc)

        print(
            f"Phase C dataset: {len(self.samples)} samples fit 768-token budget "
            f"({skipped_budget} over budget, {skipped_corrupt} corrupt/missing, "
            f"{num_categories} categories)."
        )
        if not self.samples:
            raise ValueError("No samples fit the compact token budget.")

    def __len__(self):
        return len(self.samples)

    def _remove_sample(self, img_path: str, label_path: str) -> None:
        self.samples = [(i, l) for i, l in self.samples if i != img_path]

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = min(64, len(self.samples))

        while attempts < max_attempts:
            if not self.samples:
                raise RuntimeError("All samples were removed due to corruption.")
            idx = idx % len(self.samples)
            img_path, label_path = self.samples[idx]

            try:
                image = Image.open(img_path)
                image.verify()
                image = Image.open(img_path).convert("RGB")
                global_img = image.resize((384, 384))
                img_768 = image.resize((768, 768))
                tl = img_768.crop((0, 0, 384, 384))
                tr = img_768.crop((384, 0, 768, 384))
                bl = img_768.crop((0, 384, 384, 768))
                br = img_768.crop((384, 384, 768, 768))
                pixel_values = self.processor(
                    images=[global_img, tl, tr, bl, br], return_tensors="pt"
                ).pixel_values

                with open(label_path, encoding="utf-8", errors="replace") as f:
                    verbose_obj = json.load(f)

                if random.random() < 0.05:
                    user_prompt = CLASSIFY_PROMPT
                    target_json = verbose_obj.get("chart_type", "unknown")
                else:
                    user_prompt = EXTRACTION_PROMPT
                    target_json = compact_json_string(verbose_obj)

            except Exception as e:
                print(f"\nWARNING: Removing corrupt sample {img_path}: {e}")
                _delete_corrupt_pair(img_path, label_path, e)
                self._remove_sample(img_path, label_path)
                attempts += 1
                continue

            full_text = user_prompt + target_json + self.tokenizer.eos_token
            encoded = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=768,
                return_tensors="pt",
            )
            input_ids = encoded.input_ids.squeeze(0)
            attention_mask = encoded.attention_mask.squeeze(0)
            labels = input_ids.clone()
            prompt_len = self.tokenizer(
                user_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.shape[1]
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        raise RuntimeError(f"Too many consecutive corrupt samples near index {idx}.")


def main():
    IMAGE_DIR = r"C:\sem4\KMVision-1 Data\dataset\train_1\images"
    LABEL_DIR = r"C:\sem4\KMVision-1 Data\dataset\train_1\labels"
    PHASE_B_FINAL = r"checkpoints/phase_b/final"
    CHECKPOINT_DIR = r"checkpoints/phase_c/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 16
    LEARNING_RATE = 5e-5
    SUBSET_SIZE = 30000
    MAX_GLOBAL_STEPS = 2000
    CHECKPOINT_EVERY = 250

    device = torch.device("cuda:0")
    print(f"Phase C on device: {device}")
    try:
        torch.empty(1, device=device)
    except Exception as e:
        print(f"FATAL CUDA ERROR: {e}")
        return

    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading ClinicalMicroVLM + Phase B final adapter...")
    model = ClinicalMicroVLM(bnb_config=bnb_config)
    model.vision_encoder.requires_grad_(False)

    if not os.path.isdir(PHASE_B_FINAL):
        print(f"FATAL: Phase B checkpoint missing at {PHASE_B_FINAL}")
        return

    model.llm = PeftModel.from_pretrained(model.llm, PHASE_B_FINAL, is_trainable=True)
    model.llm.gradient_checkpointing_enable()
    model.llm.config.use_cache = False
    model.projector.requires_grad_(True)

    projector_path = os.path.join(PHASE_B_FINAL, "projector_weights.pth")
    model.projector.load_state_dict(torch.load(projector_path, map_location=device))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    model.vision_encoder = model.vision_encoder.to(device)
    model.projector = model.projector.to(device)

    optimizer = bnb.optim.PagedAdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    dataset = CompactChartDataset(
        IMAGE_DIR, LABEL_DIR, processor, tokenizer, max_samples=SUBSET_SIZE, seed=42
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    total_steps = min(len(dataloader) // GRAD_ACCUM_STEPS, MAX_GLOBAL_STEPS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )

    print(f"\n--- Phase C: compact JSON, up to {MAX_GLOBAL_STEPS} optimizer steps ---")
    print(f"Checkpoints every {CHECKPOINT_EVERY} steps -> {CHECKPOINT_DIR}")
    model.train()
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    global_step = 0
    last_loss_val = 0.0

    try:
        progress_bar = tqdm(dataloader, desc="Phase C")
        for step, batch in enumerate(progress_bar):
            if global_step >= MAX_GLOBAL_STEPS:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if step == 0:
                print(f"\n--- DIAGNOSTICS: seq_len={input_ids.shape[1]} + 3645 image tokens ---")

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss_val = loss.item() * GRAD_ACCUM_STEPS
            last_loss_val = loss_val
            loss.backward()
            del outputs, loss

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                torch.cuda.empty_cache()

                if global_step % CHECKPOINT_EVERY == 0:
                    saved = _save_checkpoint(model, CHECKPOINT_DIR, global_step, loss_val)
                    print(f"\n[Step {global_step}] Checkpoint saved -> {saved}")

                if os.path.exists("save_now.txt"):
                    saved = _save_checkpoint(
                        model,
                        CHECKPOINT_DIR,
                        global_step,
                        loss_val,
                        folder_name=f"manual_step_{global_step}",
                    )
                    os.remove("save_now.txt")
                    print(f"\n[TRIGGER] Manual checkpoint -> {saved}")

            if step == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"--- DIAGNOSTICS: Max VRAM allocated {mem:.2f} GB ---")

            vram = torch.cuda.memory_reserved() / 1024**3
            progress_bar.set_postfix({"loss": loss_val, "gstep": global_step, "vram": f"{vram:.1f}G"})
            del batch, pixel_values, input_ids, attention_mask, labels

    except KeyboardInterrupt:
        print("\nInterrupted — saving emergency checkpoint...")
        interrupt_dir = os.path.join(CHECKPOINT_DIR, "interrupt_checkpoint")
        os.makedirs(interrupt_dir, exist_ok=True)
        model.llm.save_pretrained(interrupt_dir)
        torch.save(model.projector.state_dict(), os.path.join(interrupt_dir, "projector_weights.pth"))
        with open(os.path.join(interrupt_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"global_step": global_step, "loss": last_loss_val, "saved_at_utc": _utc_now()}, f)
        return

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nOOM at global_step={global_step}: {e}")
        raise

    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.llm.save_pretrained(final_dir)
    torch.save(model.projector.state_dict(), os.path.join(final_dir, "projector_weights.pth"))
    with open(os.path.join(final_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"global_step": global_step, "loss": last_loss_val, "saved_at_utc": _utc_now()}, f)
    print(f"\nPhase C complete. Weights saved to {final_dir}")


if __name__ == "__main__":
    main()
