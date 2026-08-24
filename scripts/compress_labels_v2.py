"""Compress the KM v2 corpus (extended schema) into training labels.

Reads {root}/train_v2/labels/km/*.json (full KMChartSchema with title,
time_unit, HR, CI, p-value, at-risk table) and writes minified targets to
{root}/labels_compressed_v2/km/*.json via schema_compact.minify_chart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = r"C:\sem4\KMVision-1 Data\dataset"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=DEFAULT_ROOT)
    args = p.parse_args()

    from evaluation.schema_compact import minify_chart

    src = Path(args.dataset_root) / "train_v2" / "labels" / "km"
    dst = Path(args.dataset_root) / "labels_compressed_v2" / "km"
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.json"))
    t0 = time.time()
    n_ok = 0
    n_err = 0
    for i, lf in enumerate(files):
        try:
            obj = json.loads(lf.read_text(encoding="utf-8", errors="replace"))
            mini = minify_chart(obj)
            (dst / lf.name).write_text(
                json.dumps(mini, separators=(",", ":")), encoding="utf-8"
            )
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"ERROR {lf.name}: {exc}", flush=True)
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"DONE ok={n_ok} err={n_err} -> {dst}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
