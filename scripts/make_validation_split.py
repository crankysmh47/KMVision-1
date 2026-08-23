"""
Create the frozen 500-chart synthetic KM validation partition (Phase 0, plan §3.5).

- Samples 500 chart IDs from `testing/` via split_manifest.json with a fixed seed
  (42), disjoint from any prior eval usage.
- Writes `validation_manifest.json` to the dataset root: {"km": [chart_id, ...]}.
- Does NOT copy or move images/labels — scripts read through the manifest.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 500-chart validation manifest.")
    parser.add_argument(
        "--dataset-root",
        default=r"C:\sem4\KMVision-1 Data\dataset",
    )
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: {dataset_root}/validation_manifest.json)",
    )
    parser.add_argument(
        "--exclude-manifests", nargs="*", default=[],
        help="Manifest JSON paths whose km chart IDs must NOT be sampled "
             "(used to keep partitions disjoint).",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    manifest_path = root / "split_manifest.json"
    if not manifest_path.is_file():
        print(f"Missing split manifest: {manifest_path}")
        return 1

    split = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Manifest entries are {"image": path, "label": path}; extract chart stems.
    test_ids = set()
    val = split.get("test")
    if isinstance(val, list) and val:
        first = val[0]
        if isinstance(first, dict):
            test_ids.update(
                Path(entry["label"]).stem
                for entry in val
                if isinstance(entry, dict) and entry.get("label", "").endswith("_km.json")
            )
        elif isinstance(first, str):
            test_ids.update(Path(p).stem for p in val)
    if not test_ids:
        lbl_dir = root / "testing" / "labels" / "km"
        test_ids = {p.stem for p in lbl_dir.glob("*.json")}

    # Exclude any chart already used by prior tile holdouts (belt & suspenders).
    for holdout in ("stage2_v2_1_holdout", "stage2_train1_holdout"):
        hd = root / holdout / "labels" / "km"
        if hd.is_dir():
            for lf in hd.glob("*.json"):
                meta = json.loads(lf.read_text(encoding="utf-8")).get("_meta", {})
                sc = meta.get("source_chart")
                if sc:
                    test_ids.discard(sc)

    # Exclude charts claimed by other partitions (e.g. the validation manifest).
    excluded: set[str] = set()
    for ex_path in args.exclude_manifests:
        ex = Path(ex_path)
        if not ex.is_file():
            print(f"WARNING: exclude manifest missing: {ex}")
            continue
        data = json.loads(ex.read_text(encoding="utf-8"))
        excluded.update(data.get("categories", {}).get("km", []))
    test_ids -= excluded

    pool = sorted(test_ids)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    selected = sorted(pool[: args.n])

    out_path = Path(args.output) if args.output else root / "validation_manifest.json"
    out_path.write_text(
        json.dumps({"seed": args.seed, "source": "testing", "categories": {"km": selected}}, indent=2),
        encoding="utf-8",
    )

    # Report overlap with prior holdouts (must be 0).
    overlap = 0
    hd = root / "stage2_v2_1_holdout" / "labels" / "km"
    holdout_charts = {
        json.loads(lf.read_text(encoding="utf-8")).get("_meta", {}).get("source_chart")
        for lf in hd.glob("*.json")
    } if hd.is_dir() else set()
    overlap = len(set(selected) & holdout_charts)

    print(f"Validation manifest written: {out_path}")
    print(f"Charts selected: {len(selected)} | pool: {len(pool)} | overlap w/ stage2 holdout: {overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
