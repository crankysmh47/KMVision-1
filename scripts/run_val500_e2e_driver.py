"""Drive repeated val500 e2e_oracle passes around cumulative GPU corruption.

Each pass runs run_validation_benchmark.py --arm e2e_oracle --stop-after 20.
The runner resumes from partial JSONLs, so every pass only spends time on
charts not yet cleanly scored. Stops when unique clean coverage >= target,
after max_passes passes, or if the runner exits with an unexpected code.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "val500_e2e_batch4.log"
TARGET = 200
MAX_PASSES = 40


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def unique_clean() -> int:
    done: set[str] = set()
    for f in (ROOT / "evaluation/results/val500/e2e_oracle").glob("partial_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" not in rec and isinstance(rec.get("score"), dict):
                done.add(rec["chart"])
    return len(done)


def main() -> int:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"=== e2e_oracle accumulation loop start {utc()} "
                  f"target={TARGET} max_passes={MAX_PASSES} ===\n")
        log.flush()
        runner = ROOT / "venv" / "Scripts" / "python.exe"
        for i in range(1, MAX_PASSES + 1):
            proc = subprocess.run(
                [str(runner), str(ROOT / "scripts" / "run_validation_benchmark.py"),
                 "--arm", "e2e_oracle", "--stop-after", "20"],
                cwd=str(ROOT), capture_output=True, text=True,
            )
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-3:])
            n = unique_clean()
            log.write(f"[{utc()}] PASS{i} rc={proc.returncode} unique_clean={n}\n{tail}\n")
            log.flush()
            if n >= TARGET:
                log.write(f"ALL_DONE unique_clean={n}\n")
                break
            if proc.returncode not in (0, 3):
                log.write(f"BAD_RC={proc.returncode} aborting\n")
                return 1
            time.sleep(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
