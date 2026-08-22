"""
Evaluate a macro checkpoint on labeled real-world KM charts.

Usage:
  python eval_realworld.py --checkpoint checkpoints/realworld_macro_km/final
  python eval_realworld.py --checkpoint checkpoints/phase_c_run2_chatml/final
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval_inference import build_inputs_embeds, decompress_json, generate_extraction, load_model
from evaluation.metrics import aggregate_scores, score_extraction
from evaluation.parse_output import extract_json_from_text

REAL_ROOT = ROOT / "real_dataset"


def collect_labeled_pairs(chart_type: str = "km") -> list[tuple[str, str]]:
    image_dir = REAL_ROOT / f"images_{chart_type}"
    label_dir = REAL_ROOT / "labels" / chart_type
    pairs = []
    for label_path in sorted(label_dir.glob("*.json")):
        for ext in (".png", ".jpg"):
            img = image_dir / f"{label_path.stem}{ext}"
            if img.is_file():
                pairs.append((str(img), str(label_path)))
                break
    return pairs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/phase_c_run2_chatml/final")
    p.add_argument("--chart-type", default="km")
    p.add_argument("--output-dir", default="evaluation/results/realworld")
    args = p.parse_args()

    pairs = collect_labeled_pairs(args.chart_type)
    if not pairs:
        print(
            f"No labeled {args.chart_type} charts in real_dataset/. "
            "Run real_dataset/labeler.py first."
        )
        return 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.checkpoint, device)
    scores = []
    records = []

    for img_path, label_path in tqdm(pairs, desc="realworld_eval"):
        with open(label_path, encoding="utf-8") as f:
            gt = json.load(f)
        try:
            raw = generate_extraction(model, processor, tokenizer, img_path, device)
            parsed, _ = extract_json_from_text(raw)
            pred = decompress_json(parsed) if parsed else raw
            score = score_extraction(gt, pred)
        except Exception as exc:
            score = score_extraction(gt, "")
            raw = str(exc)
        scores.append(score)
        records.append({"image": img_path, "label": label_path, "score": score.to_dict()})

    summary = aggregate_scores(scores)
    summary["checkpoint"] = args.checkpoint
    summary["labeled_samples"] = len(pairs)
    summary["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "latest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
