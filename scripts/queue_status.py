"""Print current week-queue / training progress."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.training_checkpoint import find_latest_step_dir, load_latest_pointer, verify_step_dir

RUNS = [
    ("Run 1 (minified)", ROOT / "checkpoints" / "phase_c_run1_minified", 2000),
    ("Run 2 (ChatML)", ROOT / "checkpoints" / "phase_c_run2_chatml", 2000),
    ("Run 3 (ChatML low LR)", ROOT / "checkpoints" / "phase_c_run3_low_lr", 2000),
]


def describe_run(name: str, run_dir: Path, max_steps: int) -> None:
    final = run_dir / "final"
    if verify_step_dir(str(final)) == []:
        print(f"{name}: COMPLETE (final/)")
        return
    pointer = load_latest_pointer(str(run_dir))
    step_num, step_dir = find_latest_step_dir(str(run_dir))
    if pointer:
        gs = int(pointer.get("global_step", 0))
        ms = int(pointer.get("micro_step", 0))
        max_s = int(pointer.get("max_global_steps", max_steps))
        weights = pointer.get("weights_step_dir") or pointer.get("step_dir", "")
        state = "yes" if pointer.get("has_training_state") else "no"
        print(f"{name}: IN PROGRESS  global_step={gs}/{max_s}  micro_step={ms}  training_state={state}")
        print(f"  weights: {weights}")
        return
    if step_dir:
        print(f"{name}: PARTIAL  step_dir={step_dir} (no latest.json yet)")
        return
    print(f"{name}: NOT STARTED")


def main() -> None:
    print("=== KMVision queue status ===\n")
    for name, path, max_steps in RUNS:
        describe_run(name, path, max_steps)
    alt = ROOT / "checkpoints" / "phase_c_run2_chatml_resume_from_500"
    if alt.is_dir():
        step_num, step_dir = find_latest_step_dir(str(alt))
        if step_dir:
            eff = 500 + (step_num or 0)
            print(f"\nLegacy Run 2 resume dir: {step_dir} (effective ~{eff} steps)")
    log = ROOT / "logs" / "week_queue" / "week_queue.log"
    if log.is_file():
        print(f"\nMain log: {log}")


if __name__ == "__main__":
    main()
