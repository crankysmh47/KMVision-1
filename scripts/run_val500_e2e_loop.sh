#!/usr/bin/env bash
# Accumulate Stage-2 E2E-oracle scores on the frozen val500 split.
# Works around cumulative RTX 5060 Ti GPU corruption: each pass runs at most
# --stop-after 20 fresh charts, then exits (exit 3 on AcceleratorError);
# resume skips already-scored charts. Stops when unique clean coverage hits
# the target or after MAX_PASSES passes. Log: logs/val500_e2e_batch4.log
set +m
cd /c/sem4/KMVision-1
LOG=logs/val500_e2e_batch4.log
TARGET=200
MAX_PASSES=40
echo "=== e2e_oracle accumulation loop start $(date -u +%FT%TZ) target=$TARGET ===" >> "$LOG"
for i in $(seq 1 $MAX_PASSES); do
  venv/Scripts/python.exe scripts/run_validation_benchmark.py --arm e2e_oracle --stop-after 20 >> "$LOG" 2>&1
  rc=$?
  n=$(venv/Scripts/python.exe - <<'PY'
import json
from pathlib import Path
done = set()
for f in Path("evaluation/results/val500/e2e_oracle").glob("partial_*.jsonl"):
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" not in r and isinstance(r.get("score"), dict):
            done.add(r["chart"])
print(len(done))
PY
)
  echo "PASS$i rc=$rc unique_clean=$n" >> "$LOG"
  if [ "$n" -ge "$TARGET" ]; then echo "ALL_DONE unique_clean=$n" >> "$LOG"; break; fi
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then echo "BAD_RC=$rc aborting" >> "$LOG"; break; fi
  sleep 5
done
