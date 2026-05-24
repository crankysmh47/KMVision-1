"""
Compress verbose chart labels to minified JSON on disk.

Reads category subfolders from --input-dir and writes parallel structure to --output-dir.
Kaplan-Meier arms use separate t/s/c arrays (time_points, survival_probabilities, censoring_ticks).
Other chart types use evaluation.schema_compact.minify_chart.

Usage:
  python scripts/compress_labels.py
  python scripts/compress_labels.py --input-dir "C:\\...\\train_1\\labels" --output-dir "C:\\...\\labels_compressed"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from evaluation.schema_compact import (
    subsample_km_coordinates,
    _cap_evenly,
    _dedupe_sorted_floats,
    _round_float,
    minify_chart,
)

DEFAULT_DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"
DEFAULT_MAX_POINTS = 20


def compress_km_arm(arm: dict, *, max_points: int) -> dict:
    """Step-aware subsample, then split coordinates into t/s arrays; c -> censoring_ticks."""
    coords = subsample_km_coordinates(arm.get("coordinates", []))
    coords = [[round(float(t), 2), round(float(s), 4)] for t, s in coords]
    coords = _cap_evenly(coords, max_points)
    times = [pt[0] for pt in coords]
    surv = [pt[1] for pt in coords]
    censors = _dedupe_sorted_floats(arm.get("censoring_ticks", []))
    censors = [_round_float(v) for v in censors]
    return {
        "id": arm.get("treatment_label", ""),
        "t": times,
        "s": surv,
        "c": censors,
    }


def compress_km_chart(obj: dict, *, max_points: int) -> dict:
    axes = obj.get("axes", {})
    x_ax, y_ax = axes.get("x", {}), axes.get("y", {})
    return {
        "ct": "km",
        "ax": {
            "x": {"l": x_ax.get("label", ""), "m": _round_float(x_ax.get("max_value", 0))},
            "y": {"l": y_ax.get("label", ""), "m": _round_float(y_ax.get("max_value", 1.0), 4)},
        },
        "a": [compress_km_arm(arm, max_points=max_points) for arm in obj.get("arms", [])],
    }


def compress_label_obj(obj: dict, *, max_km_points: int) -> dict:
    if obj.get("chart_type") == "kaplan_meier":
        return compress_km_chart(obj, max_points=max_km_points)
    if "ct" in obj and "chart_type" not in obj:
        return obj
    return minify_chart(obj)


def collect_label_files(input_dir: str) -> list[tuple[str, str]]:
    """Return list of (relative_path_from_input, absolute_path)."""
    pairs = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, input_dir)
            pairs.append((rel_path, abs_path))
    return pairs


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
        "--max-km-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Max step points per KM arm after subsampling (default 20).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input dir not found: {args.input_dir}")
        sys.exit(1)

    files = collect_label_files(args.input_dir)
    if not files:
        print(f"No JSON labels under {args.input_dir}")
        sys.exit(1)

    print(f"Compressing {len(files):,} labels")
    print(f"  from: {args.input_dir}")
    print(f"  to:   {args.output_dir}")

    written = 0
    errors = 0
    for rel_path, src_path in tqdm(files, desc="compress"):
        try:
            with open(src_path, encoding="utf-8", errors="replace") as f:
                obj = json.load(f)
            compressed = compress_label_obj(obj, max_km_points=args.max_km_points)
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
            print(f"\nERROR {src_path}: {exc}")

    print(f"Done. written={written:,} errors={errors:,}")


if __name__ == "__main__":
    main()
