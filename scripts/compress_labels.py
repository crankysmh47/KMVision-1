"""
Compress verbose chart labels to minified JSON on disk.

Uses evaluation.schema_compact.minify_chart (768-token budget caps for KM and other types).
Skips empty or invalid JSON files and logs them.

Usage:
  python scripts/compress_labels.py
  python scripts/compress_labels.py --input-dir "C:\\...\\train_1\\labels" --output-dir "C:\\...\\labels_compressed"
  python scripts/compress_labels.py --category km
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from evaluation.schema_compact import minify_chart

DEFAULT_DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"
CORRUPT_LOG = "corrupted_labels.log"


def compress_label_obj(obj: dict) -> dict:
    if "ct" in obj and "chart_type" not in obj:
        return obj
    return minify_chart(obj)


def collect_label_files(input_dir: str, *, category: str | None = None) -> list[tuple[str, str]]:
    """Return list of (relative_path_from_input, absolute_path)."""
    pairs = []
    if category:
        roots = [os.path.join(input_dir, category)]
    else:
        roots = [
            os.path.join(input_dir, name)
            for name in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, name))
        ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for walk_root, _, files in os.walk(root):
            for name in files:
                if not name.endswith(".json"):
                    continue
                abs_path = os.path.join(walk_root, name)
                rel_path = os.path.relpath(abs_path, input_dir)
                pairs.append((rel_path, abs_path))
    return pairs


def log_corrupt(path: str, error: Exception) -> None:
    line = f"{path}\t{error}\n"
    with open(CORRUPT_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress verbose JSON labels to minified format.")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(DEFAULT_DATASET_ROOT, "train_1", "labels"),
        help="Source labels tree (default: dataset/train_1/labels).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(DEFAULT_DATASET_ROOT, "labels_compressed"),
        help="Destination for compressed labels (mirrors input folder structure).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Only compress one category subfolder (e.g. km).",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=100,
        help="Exit 1 if more than this many files fail (default 100).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input dir not found: {args.input_dir}")
        sys.exit(1)

    files = collect_label_files(args.input_dir, category=args.category)
    if not files:
        print(f"No JSON labels under {args.input_dir}")
        sys.exit(1)

    print(f"Compressing {len(files):,} labels")
    print(f"  from: {args.input_dir}")
    print(f"  to:   {args.output_dir}")
    if args.category:
        print(f"  category: {args.category}")

    written = 0
    errors = 0
    for rel_path, src_path in tqdm(files, desc="compress"):
        try:
            raw = open(src_path, encoding="utf-8", errors="replace").read().strip()
            if not raw:
                raise ValueError("empty label file")
            obj = json.loads(raw)
            compressed = compress_label_obj(obj)
            out_path = os.path.join(args.output_dir, rel_path)
            if args.dry_run:
                written += 1
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(compressed, f, separators=(",", ":"))
            written += 1
        except Exception as exc:
            errors += 1
            log_corrupt(src_path, exc)
            print(f"\nERROR {src_path}: {exc}")

    print(f"Done. written={written:,} errors={errors:,}")
    if errors > args.max_errors:
        print(f"FATAL: {errors} errors exceeds --max-errors {args.max_errors}")
        sys.exit(1)
    if errors:
        print(f"Skipped {errors} corrupt file(s); see {CORRUPT_LOG}")


if __name__ == "__main__":
    main()
