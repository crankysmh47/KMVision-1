"""Generate the Phase 0.5 KM-only synthetic corpus with the extended schema.

Produces charts with title, time_unit, hazard_ratio, CI, p_value and
at_risk_table ground truth (see synth_dataset/schemas.py KMChartSchema).

Output layout: {output_root}/images/km/*.png + {output_root}/labels/km/*.json
Default output root: C:\\sem4\\KMVision-1 Data\\dataset\\train_v2

Usage:
    venv/Scripts/python.exe scripts/generate_km_v2_corpus.py --n 12000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "synth_dataset"
DEFAULT_OUTPUT_ROOT = r"C:\sem4\KMVision-1 Data\dataset\train_v2"

_generate_kwargs: dict = {}


def worker(task_idx: int) -> bool:
    from generate_km import generate_km_chart

    try:
        generate_km_chart(output_dir=_generate_kwargs.get("output_root"))
        return True
    except Exception:
        return False


def init_worker(output_root: str) -> None:
    _generate_kwargs["output_root"] = output_root


def main() -> int:
    parser = argparse.ArgumentParser(description="KM v2 corpus generation.")
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    sys.path.insert(0, str(SYNTH))
    out_root = Path(args.output_root)
    lbl_dir = out_root / "labels" / "km"

    print(f"Generating {args.n} KM v2 charts -> {out_root} using {args.procs} procs")
    t0 = time.time()
    done = 0
    with mp.Pool(processes=args.procs, maxtasksperchild=500,
                 initializer=init_worker, initargs=(str(out_root),)) as pool:
        for _ in pool.imap_unordered(worker, range(args.n), chunksize=8):
            done += 1
            if done % 500 == 0:
                rate = done / max(time.time() - t0, 1)
                print(f"{done}/{args.n} ({rate:.1f} charts/s)", flush=True)

    n_labels = len(list(lbl_dir.glob("*.json")))
    elapsed = time.time() - t0
    print(f"DONE requested={args.n} written={n_labels} "
          f"elapsed={elapsed / 60:.1f}min", flush=True)
    return 0 if n_labels >= args.n * 0.99 else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
