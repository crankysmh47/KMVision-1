"""
Stage 2 micro-detail extractor: single 384x384 tile -> dense arm-local JSON.

Architecture: one SigLIP crop (729 visual tokens), NOT 5-crop AnyRes pooling.
Initialization: fresh LoRA + Phase A projector only (never Phase B/C or prior stage2 weights).

See docs/STAGE2_DECISIONS.md.

Usage:
  # v2.1 sanity (normalized local coords, prefix-masked loss):
  python train_stage2.py --max_global_steps 500 --no_auto_resume
  # full v2.1 run after sanity passes:
  python train_stage2.py --max_global_steps 3000 --no_auto_resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.schema_compact import build_training_text
from model import ClinicalMicroVLM
from stage2_common import mask_len_through_json_prefix, stage2_target_json, stage2_user_prompt
from scripts.training_checkpoint import (
    load_training_state,
    resolve_resume,
    save_latest_pointer,
    save_progress,
    save_training_state,
    step_dir_for,
    verify_step_dir,
)
from scripts.training_lock import acquire_lock, release_lock

DEFAULT_DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"
DEFAULT_TILE_SUBDIR = "stage2_v2_1"
PROJECTOR_INIT = ROOT / "checkpoints" / "checkpoints_projector" / "projector_weights.pth"

NUM_IMAGE_TOKENS = 729  # single 384 patch, 27x27


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def target_json_from_label(label_obj: dict) -> str:
    """Training target: flat_xy points/censors, excludes _meta."""
    return stage2_target_json(label_obj)


class Stage2TileDataset(Dataset):
    """Loads pre-generated 384x384 tiles from dataset/stage2/."""

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        processor,
        tokenizer,
        *,
        max_samples: int = 50000,
        seed: int = 42,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.samples: list[tuple[str, str]] = []

        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Missing {image_dir} or {label_dir}")

        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".json"):
                continue
            stem = os.path.splitext(label_file)[0]
            for ext in (".png", ".jpg"):
                img_path = os.path.join(image_dir, stem + ext)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, os.path.join(label_dir, label_file)))
                    break

        rng = random.Random(seed)
        rng.shuffle(self.samples)
        self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(
                f"No tile pairs in {image_dir}. Run scripts/generate_stage2_tiles.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label_path = self.samples[idx % len(self.samples)]
        image = Image.open(img_path).convert("RGB")
        if image.size != (384, 384):
            image = image.resize((384, 384), Image.Resampling.LANCZOS)

        # Single crop -> (1, C, H, W) for ClinicalMicroVLM (DECISIONS §7).
        pixel_values = self.processor(images=[image], return_tensors="pt").pixel_values

        with open(label_path, encoding="utf-8") as f:
            label_obj = json.load(f)

        arm_id = label_obj.get("arm_id", "unknown")
        user_prompt = stage2_user_prompt(arm_id)
        target_json = target_json_from_label(label_obj)

        full_text = build_training_text(
            user_prompt, target_json, self.tokenizer, use_chatml=True
        )
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        labels = input_ids.clone()
        mask_len = mask_len_through_json_prefix(user_prompt, arm_id, self.tokenizer)
        labels[:mask_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _save_checkpoint(
    model,
    checkpoint_dir: str,
    global_step: int,
    micro_step: int,
    loss_val: float,
    optimizer,
    scheduler,
    args_dict: dict,
) -> str:
    step_dir = step_dir_for(checkpoint_dir, global_step)
    os.makedirs(step_dir, exist_ok=True)
    model.llm.save_pretrained(step_dir)
    torch.save(model.projector.state_dict(), os.path.join(step_dir, "projector_weights.pth"))
    save_training_state(
        os.path.join(step_dir, "training_state.pt"),
        global_step=global_step,
        micro_step=micro_step,
        optimizer=optimizer,
        scheduler=scheduler,
        last_loss=loss_val,
        args_dict=args_dict,
    )
    meta = {
        "global_step": global_step,
        "micro_step": micro_step,
        "loss": loss_val,
        "saved_at_utc": _utc_now(),
        "num_image_tokens": NUM_IMAGE_TOKENS,
    }
    with open(os.path.join(step_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    errors = verify_step_dir(step_dir)
    if errors:
        raise RuntimeError(f"Incomplete checkpoint: {errors}")
    save_latest_pointer(
        checkpoint_dir,
        global_step=global_step,
        step_dir=step_dir,
        micro_step=micro_step,
        max_global_steps=int(args_dict.get("max_global_steps", 3000)),
        has_training_state=True,
        init_checkpoint=str(args_dict.get("projector_init", "")),
    )
    return step_dir


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 tile micro-extractor training.")
    p.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--label_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="checkpoints/stage2_v2_1")
    p.add_argument("--projector_init", type=str, default=str(PROJECTOR_INIT))
    p.add_argument("--subset_size", type=int, default=50000)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument(
        "--max_global_steps",
        type=int,
        default=500,
        help="Default 500 for v2 sanity check; use 3000 for full training.",
    )
    p.add_argument("--checkpoint_every", type=int, default=250)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--no_auto_resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        acquire_lock(pid=os.getpid(), label=f"train_stage2 {args.output_dir}")
    except RuntimeError as exc:
        print(f"FATAL: {exc}")
        return

    try:
        _train_body(args)
    finally:
        release_lock(pid=os.getpid())


def _train_body(args):
    tile_root = os.path.join(args.dataset_root, DEFAULT_TILE_SUBDIR)
    image_dir = args.image_dir or os.path.join(tile_root, "images", "km")
    label_dir = args.label_dir or os.path.join(tile_root, "labels", "km")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0")
    print(f"Stage 2 training on {device}")
    print(f"  tiles:      {image_dir}")
    print(f"  labels:     {label_dir}")
    print(f"  output:     {args.output_dir}")
    print(f"  projector:  {args.projector_init}")
    print(f"  image tok:  {NUM_IMAGE_TOKENS} (single 384 crop)")

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

    resume_info = None
    if args.auto_resume and not args.no_auto_resume:
        resume_info = resolve_resume(args.output_dir, None)
        if resume_info:
            print(
                "WARNING: --auto_resume loaded prior weights. "
                "For a fresh v2 run, use a new --output_dir or --no_auto_resume."
            )

    model = ClinicalMicroVLM(bnb_config=bnb_config)
    model.vision_encoder.requires_grad_(False)

    if resume_info and os.path.isdir(resume_info["step_dir"]):
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, resume_info["step_dir"], is_trainable=True)
        model.projector.load_state_dict(
            torch.load(
                os.path.join(resume_info["step_dir"], "projector_weights.pth"),
                map_location=device,
            )
        )
        print(f"Resumed adapter from {resume_info['step_dir']}")
    else:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        model.llm = get_peft_model(model.llm, lora_config)
        if os.path.isfile(args.projector_init):
            model.projector.load_state_dict(
                torch.load(args.projector_init, map_location=device)
            )
            print(f"Loaded Phase A projector from {args.projector_init}")
        else:
            print(f"WARNING: projector init missing at {args.projector_init}")

    model.llm.gradient_checkpointing_enable()
    model.llm.config.use_cache = False
    model.projector.requires_grad_(True)
    model.vision_encoder = model.vision_encoder.to(device)
    model.projector = model.projector.to(device)

    dataset = Stage2TileDataset(
        image_dir,
        label_dir,
        processor,
        tokenizer,
        max_samples=args.subset_size,
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    optimizer = bnb.optim.PagedAdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    total_steps = min(len(dataloader) // args.grad_accum_steps, args.max_global_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )

    global_step = int(resume_info["global_step"]) if resume_info else 0
    start_micro = int(resume_info["micro_step"]) if resume_info else 0
    if resume_info and resume_info.get("has_training_state"):
        load_training_state(
            resume_info["training_state_path"],
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    args_dict = vars(args).copy()
    model.train()
    optimizer.zero_grad()
    last_loss = 0.0
    weights_step_dir = resume_info["step_dir"] if resume_info else args.output_dir

    print(f"--- Stage 2: steps {global_step} -> {args.max_global_steps} ---")
    progress = tqdm(dataloader, desc="Stage2")
    try:
        for step, batch in enumerate(progress):
            if step < start_micro:
                continue
            if global_step >= args.max_global_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum_steps
            last_loss = loss.item() * args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                torch.cuda.empty_cache()

                if global_step % args.checkpoint_every == 0:
                    weights_step_dir = _save_checkpoint(
                        model,
                        args.output_dir,
                        global_step,
                        step + 1,
                        last_loss,
                        optimizer,
                        scheduler,
                        args_dict,
                    )
                    print(f"\n[Step {global_step}] -> {weights_step_dir}", flush=True)

                print(
                    f"PROGRESS global_step={global_step}/{args.max_global_steps} "
                    f"loss={last_loss:.4f}",
                    flush=True,
                )
                save_progress(
                    args.output_dir,
                    global_step=global_step,
                    micro_step=step + 1,
                    weights_step_dir=weights_step_dir,
                    max_global_steps=args.max_global_steps,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    last_loss=last_loss,
                    args_dict=args_dict,
                )

            progress.set_postfix({"loss": f"{last_loss:.3f}", "gstep": global_step})

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.llm.save_pretrained(final_dir)
    torch.save(model.projector.state_dict(), os.path.join(final_dir, "projector_weights.pth"))
    save_latest_pointer(
        args.output_dir,
        global_step=global_step,
        step_dir=final_dir,
        micro_step=0,
        max_global_steps=args.max_global_steps,
        has_training_state=False,
        init_checkpoint=args.projector_init,
    )
    print(f"Stage 2 complete -> {final_dir}")


if __name__ == "__main__":
    main()
