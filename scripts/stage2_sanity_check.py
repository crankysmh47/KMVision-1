"""
Quick Stage 2 sanity check: run inference on a few holdout tiles and report
whether outputs are strict valid JSON with arm_id, points, and censors.

Usage:
  python scripts/stage2_sanity_check.py
  python scripts/stage2_sanity_check.py --checkpoint checkpoints/stage2_v2/final --max-samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_stage2 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_ROOT,
    generate_tile_json,
    load_model,
    load_tile_pairs,
    parse_stage2_output,
    resolve_checkpoint,
    sample_pairs,
)
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 2 JSON sanity check on holdout tiles.")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    p.add_argument("--holdout-dir", default=None)
    p.add_argument("--max-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument(
        "--force-json-prefix",
        action="store_true",
        help='Pre-fill assistant with {"arm_id": "<id>", "points": [ before generation.',
    )
    args = p.parse_args()

    root = Path(args.dataset_root)
    holdout = Path(args.holdout_dir or root / "stage2_v2_1_holdout")
    image_dir = holdout / "images" / "km"
    label_dir = holdout / "labels" / "km"
    pairs = sample_pairs(
        load_tile_pairs(image_dir, label_dir),
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not pairs:
        print(f"No holdout tiles in {holdout}")
        return 1

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(checkpoint, device)
    strict_ok = 0
    print(f"Checkpoint: {checkpoint}  |  tiles: {len(pairs)}")
    if args.force_json_prefix:
        print("  force_json_prefix: ON")
    print()

    for i, (img_path, label_path) in enumerate(pairs):
        with open(label_path, encoding="utf-8") as f:
            gt = json.load(f)
        arm_id = gt.get("arm_id", "unknown")
        raw = generate_tile_json(
            model,
            processor,
            tokenizer,
            img_path,
            arm_id,
            device,
            max_new_tokens=args.max_new_tokens,
            force_json_prefix=args.force_json_prefix,
        )
        pred = parse_stage2_output(raw)
        ok = (
            pred is not None
            and "arm_id" in pred
            and "points" in pred
            and "censors" in pred
        )
        strict_ok += int(ok)
        print(f"--- tile {i + 1}: {img_path.name} ---")
        print(f"  strict_json: {'PASS' if ok else 'FAIL'}")
        if pred:
            print(f"  arm_id: {pred.get('arm_id')!r}")
            print(f"  n_points: {len(pred.get('points', []))}  n_censors: {len(pred.get('censors', []))}")
        print(f"  preview: {raw[:200]!r}")
        print()

    rate = strict_ok / len(pairs)
    print(f"Strict JSON pass rate: {strict_ok}/{len(pairs)} ({rate:.0%})")
    if rate >= 0.8:
        print("Sanity check PASSED — ready for full 3000-step training.")
        return 0
    print("Sanity check FAILED — inspect previews before a full run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
