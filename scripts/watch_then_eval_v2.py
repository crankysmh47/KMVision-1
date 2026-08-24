"""Wait for v2_sft_run1/final to appear, then launch the validation_v2 eval."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "checkpoints" / "v2_sft_run1" / "final"
LOG = ROOT / "logs" / "eval_valv2.log"


def main() -> int:
    print(f"waiting for {FINAL}", flush=True)
    while True:
        if (FINAL / "checkpoint_meta.json").is_file() and \
                (FINAL / "projector_weights.pth").is_file():
            break
        time.sleep(60)
    print("final checkpoint detected; cooling down 120s", flush=True)
    time.sleep(120)
    with open(LOG, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(ROOT / "venv" / "Scripts" / "python.exe"),
             str(ROOT / "scripts" / "eval_v2_fields.py"),
             "--ckpt", "checkpoints/v2_sft_run1/final"],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
        )
    print(f"eval finished rc={proc.returncode}", flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
