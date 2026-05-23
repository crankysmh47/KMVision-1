"""
Measure tokenizer length of verbose vs compact chart labels.

Usage: python scripts/audit_label_token_lengths.py [--sample 500]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from evaluation.schema_compact import compact_json_string

DEFAULT_LABEL_DIR = r"C:\sem4\KMVision-1 Data\dataset\train_1\labels"
PROMPT = "\nExtract the underlying data from this clinical chart in strict JSON format.\n"
MAX_LENGTH = 768


def collect_json_paths(label_dir: str) -> list:
    paths = []
    for root, _, files in os.walk(label_dir):
        for f in files:
            if f.endswith(".json"):
                paths.append(os.path.join(root, f))
    return paths


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True).input_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    paths = collect_json_paths(args.label_dir)
    rng = random.Random(args.seed)
    if len(paths) > args.sample:
        paths = rng.sample(paths, args.sample)

    stats = {}
    for path in paths:
        cat = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        verbose = json.dumps(obj, separators=(",", ":"))
        compact = compact_json_string(obj)
        v_len = token_len(tokenizer, PROMPT + verbose + tokenizer.eos_token)
        c_len = token_len(tokenizer, PROMPT + compact + tokenizer.eos_token)
        if cat not in stats:
            stats[cat] = {"verbose": [], "compact": []}
        stats[cat]["verbose"].append(v_len)
        stats[cat]["compact"].append(c_len)

    print(f"Sampled {len(paths)} labels from {args.label_dir}\n")
    print(f"Budget: max_length={MAX_LENGTH} (prompt + JSON + eos)\n")
    for cat, d in sorted(stats.items()):
        v = d["verbose"]
        c = d["compact"]
        print(f"=== {cat} (n={len(v)}) ===")
        print(f"  verbose  mean={sum(v)/len(v):.0f}  max={max(v)}  >768={sum(1 for x in v if x>768)}")
        print(f"  compact  mean={sum(c)/len(c):.0f}  max={max(c)}  >768={sum(1 for x in c if x>768)}")
        print()


if __name__ == "__main__":
    main()
