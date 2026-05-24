"""
Verify compressed labels fit the 768-token training budget before week queue training.

Usage:
  python scripts/check_compress_gate.py
  python scripts/check_compress_gate.py --label-dir "C:\\...\\labels_compressed"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoTokenizer

DEFAULT_GATES = os.path.join(ROOT, "config", "eval_gates.json")
DEFAULT_LABEL_DIR = r"C:\sem4\KMVision-1 Data\dataset\labels_compressed"
PROMPT = "\nExtract the underlying data from this clinical chart in strict JSON format.\n"
MAX_LENGTH = 768


def collect_by_category(label_dir: str) -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = {}
    for root, _, files in os.walk(label_dir):
        cat = os.path.basename(root)
        if cat == os.path.basename(label_dir):
            continue
        for name in files:
            if name.endswith(".json"):
                by_cat.setdefault(cat, []).append(os.path.join(root, name))
    return by_cat


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True).input_ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--gates", default=DEFAULT_GATES)
    args = parser.parse_args()

    if not os.path.isdir(args.label_dir):
        print(f"ERROR: label dir missing: {args.label_dir}")
        print("Run scripts/compress_labels.py first.")
        return 1

    with open(args.gates, encoding="utf-8") as f:
        cfg = json.load(f)["compress_gate"]

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    by_cat = collect_by_category(args.label_dir)
    rng = random.Random(42)
    failures = []

    for cat in cfg.get("categories_required", []):
        if cat not in by_cat:
            failures.append(f"category '{cat}' missing from {args.label_dir}")
            continue
        paths = by_cat[cat]
        sample_n = min(cfg.get("sample_size", 200), len(paths))
        sample = rng.sample(paths, sample_n)
        over = 0
        for path in sample:
            with open(path, encoding="utf-8") as f:
                body = f.read()
            text = PROMPT + body + tokenizer.eos_token
            if token_len(tokenizer, text) > MAX_LENGTH:
                over += 1
        frac = over / sample_n if sample_n else 1.0
        max_frac = cfg.get("max_fraction_over_768", 0.05)
        print(f"  {cat}: {over}/{sample_n} over 768 ({frac:.1%}), max allowed {max_frac:.1%}")
        if frac > max_frac:
            failures.append(f"{cat}: {frac:.1%} over budget > {max_frac:.1%}")

    if failures:
        print("COMPRESS GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("COMPRESS GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
