"""
Fine-tune the production macro model (Phase C Run 2) on labeled real-world KM charts.

Expects verbose JSON labels from real_dataset/labeler.py paired with images in
real_dataset/images_km/ (or images_{type}/).

Usage:
  python train_realworld.py --chart-type km --max-global-steps 500
  python train_realworld.py --init-checkpoint checkpoints/phase_c_run2_chatml/final
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
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.image_preprocess import pixel_values_from_path
from evaluation.schema_compact import build_training_text, minify_chart, precompressed_fits_token_budget
from model import ClinicalMicroVLM
from scripts.training_lock import acquire_lock, release_lock

DEFAULT_INIT = "checkpoints/phase_c_run2_chatml/final"
REAL_ROOT = ROOT / "real_dataset"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RealWorldChartDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        processor,
        tokenizer,
        *,
        max_length: int = 768,
        seed: int = 42,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[tuple[str, str]] = []

        for label_path in sorted(label_dir.glob("*.json")):
            for ext in (".png", ".jpg"):
                img = image_dir / f"{label_path.stem}{ext}"
                if img.is_file():
                    self.samples.append((str(img), str(label_path)))
                    break

        rng = random.Random(seed)
        rng.shuffle(self.samples)
        if not self.samples:
            raise ValueError(
                f"No labeled pairs in {image_dir}. Label charts with real_dataset/labeler.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label_path = self.samples[idx]
        with open(label_path, encoding="utf-8") as f:
            verbose = json.load(f)
        compact = minify_chart(verbose)
        target_json = json.dumps(compact, separators=(",", ":"))
        user_prompt = "\nExtract the underlying data from this clinical chart in strict JSON format.\n"
        full_text = build_training_text(user_prompt, target_json, self.tokenizer, use_chatml=True)

        if not precompressed_fits_token_budget(full_text, self.tokenizer, self.max_length):
            target_json = json.dumps({"ct": "km", "a": compact.get("a", [])[:2]}, separators=(",", ":"))
            full_text = build_training_text(user_prompt, target_json, self.tokenizer, use_chatml=True)

        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        pixel_values = pixel_values_from_path(img_path, self.processor)
        return {
            "pixel_values": pixel_values,
            "input_ids": encoded.input_ids.squeeze(0),
            "attention_mask": encoded.attention_mask.squeeze(0),
            "labels": encoded.input_ids.squeeze(0).clone(),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune macro model on real-world labeled charts.")
    p.add_argument("--chart-type", default="km", choices=["km", "forest", "wf"])
    p.add_argument("--init-checkpoint", default=DEFAULT_INIT)
    p.add_argument("--output-dir", default="checkpoints/realworld_macro_km")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-global-steps", type=int, default=500)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--max-length", type=int, default=768)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        acquire_lock(pid=os.getpid(), label=f"train_realworld {args.output_dir}")
    except RuntimeError as exc:
        print(f"FATAL: {exc}")
        return 1

    chart_type = args.chart_type
    image_dir = REAL_ROOT / f"images_{chart_type}"
    label_dir = REAL_ROOT / "labels" / chart_type
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = RealWorldChartDataset(
        image_dir, label_dir, processor, tokenizer, max_length=args.max_length
    )
    print(f"Real-world samples: {len(dataset)} from {image_dir}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = ClinicalMicroVLM(bnb_config=bnb_config)
    model.vision_encoder.requires_grad_(False)
    model.vision_encoder = model.vision_encoder.to(device)
    model.projector = model.projector.to(device)
    model.projector.load_state_dict(
        torch.load(os.path.join(args.init_checkpoint, "projector_weights.pth"), map_location=device)
    )
    model.llm = PeftModel.from_pretrained(model.llm, args.init_checkpoint, is_trainable=True)
    model = model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    total_steps = min(len(dataloader) // args.grad_accum_steps, args.max_global_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    global_step = 0
    last_loss = 0.0
    for step, batch in enumerate(tqdm(dataloader, desc="realworld")):
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

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.llm.save_pretrained(final_dir)
    torch.save(model.projector.state_dict(), os.path.join(final_dir, "projector_weights.pth"))
    meta = {"saved_at": _utc_now(), "samples": len(dataset), "steps": global_step, "loss": last_loss}
    with open(os.path.join(final_dir, "realworld_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved real-world fine-tune -> {final_dir} (loss={last_loss:.4f})")
    release_lock(pid=os.getpid())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
