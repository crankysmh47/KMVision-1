"""Partition the KM v2 corpus into TRAIN / VALIDATION / FROZEN TEST.

Reads chart IDs from {root}/train_v2/labels/km/*.json and writes:
  validation_v2_manifest.json   seed 42, n=500 (development decisions)
  frozen_test_v2_manifest.json  seed 777, n=500 (formal milestones only)
  train_v2_train_ids.txt        everything else (training input list)

Validation and frozen test are mutually exclusive; training must exclude both.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_ROOT = r"C:\sem4\KMVision-1 Data\dataset"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=DEFAULT_ROOT)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--seed-val", type=int, default=42)
    p.add_argument("--seed-test", type=int, default=777)
    args = p.parse_args()

    root = Path(args.dataset_root)
    lbl_dir = root / "train_v2" / "labels" / "km"
    pool = sorted(p.stem for p in lbl_dir.glob("*.json"))
    if len(pool) < args.n_val + args.n_test:
        print(f"ERROR: corpus has only {len(pool)} charts; "
              f"need >= {args.n_val + args.n_test}")
        return 1

    rng_val = random.Random(args.seed_val)
    shuffled = pool[:]
    rng_val.shuffle(shuffled)
    val_ids = sorted(shuffled[: args.n_val])

    rest = [c for c in shuffled if c not in set(val_ids)]
    rng_test = random.Random(args.seed_test)
    rng_test.shuffle(rest)
    test_ids = sorted(rest[: args.n_test])

    train_ids = sorted(set(pool) - set(val_ids) - set(test_ids))

    (root / "validation_v2_manifest.json").write_text(json.dumps({
        "seed": args.seed_val, "source": "train_v2",
        "categories": {"km": val_ids}}, indent=2), encoding="utf-8")
    (root / "frozen_test_v2_manifest.json").write_text(json.dumps({
        "seed": args.seed_test, "source": "train_v2",
        "categories": {"km": test_ids}}, indent=2), encoding="utf-8")
    (root / "train_v2_train_ids.txt").write_text(
        "\n".join(train_ids) + "\n", encoding="utf-8")

    assert not (set(val_ids) & set(test_ids))
    print(f"corpus={len(pool)} | val={len(val_ids)} (seed {args.seed_val}) | "
          f"frozen_test={len(test_ids)} (seed {args.seed_test}) | "
          f"train={len(train_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
