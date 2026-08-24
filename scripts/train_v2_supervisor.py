"""Supervisor: keep train_v2_sft.py running across driver-fault process deaths.

The nvlddmkm driver (591.86) intermittently kills the training process
(CUBLAS internal errors / silent native deaths). Auto-resume restores from
the last checkpoint, so this loop simply relaunches until final/ exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "train_v2_supervisor.log"
FINAL = ROOT / "checkpoints" / "v2_sft_run1" / "final"
MAX_ATTEMPTS = 20


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"=== supervisor start {utc()} ===\n")
        log.flush()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if (FINAL / "checkpoint_meta.json").is_file():
                log.write(f"[{utc()}] final checkpoint present, done\n")
                return 0
            proc = subprocess.run(
                [str(ROOT / "venv" / "Scripts" / "python.exe"),
                 str(ROOT / "train_v2_sft.py")],
                cwd=str(ROOT), env=env,
                stdout=open(ROOT / "logs" / f"train_v2_sft_att{attempt}.log", "w"),
                stderr=subprocess.STDOUT,
            )
            log.write(f"[{utc()}] attempt {attempt} rc={proc.returncode}\n")
            log.flush()
            if (FINAL / "checkpoint_meta.json").is_file():
                log.write(f"[{utc()}] final checkpoint present after attempt "
                          f"{attempt}, done\n")
                return 0
            if proc.returncode == 0:
                log.write(f"[{utc()}] rc=0 without final; stopping\n")
                return 1
            time.sleep(30)
    return 1


if __name__ == "__main__":
    sys.exit(main())
