"""Unified week queue: single log, auto-resume, checkpoint-aware."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.training_checkpoint import (
    find_latest_step_dir,
    load_latest_pointer,
    save_latest_pointer,
    verify_step_dir,
)
from scripts.training_lock import acquire_lock, clear_stale_lock, read_lock, release_lock, stale_lock_message

LOG_DIR = ROOT / "logs" / "week_queue"
MAIN_LOG = LOG_DIR / "week_queue.log"
DATASET_ROOT = Path(r"C:\sem4\KMVision-1 Data\dataset")

RUN1_OUT = ROOT / "checkpoints" / "phase_c_run1_minified"
RUN2_OUT = ROOT / "checkpoints" / "phase_c_run2_chatml"
RUN2_ALT = ROOT / "checkpoints" / "phase_c_run2_chatml_resume_from_500"
RUN3_OUT = ROOT / "checkpoints" / "phase_c_run3_low_lr"
PHASE_B = ROOT / "checkpoints" / "phase_b" / "final"

EVAL1 = ROOT / "evaluation" / "results" / "run1_minified"
EVAL2 = ROOT / "evaluation" / "results" / "run2_chatml"
EVAL3 = ROOT / "evaluation" / "results" / "run3_low_lr"


class QueueLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def line(self, msg: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{stamp}] {msg}"
        print(text, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def run_cmd(log: QueueLogger, title: str, cmd: list[str], *, cwd: Path = ROOT) -> int:
    log.line(f"=== {title} ===")
    log.line("CMD: " + " ".join(cmd))
    with open(MAIN_LOG, "a", encoding="utf-8") as out:
        out.write(f"\n--- {title} ---\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        log.line(f"FAILED ({proc.returncode}): {title}")
    else:
        log.line(f"OK: {title}")
    return proc.returncode


def final_exists(run_dir: Path) -> bool:
    final = run_dir / "final"
    return verify_step_dir(str(final)) == []


def gate_passed(eval_dir: Path, stage: str) -> bool:
    gate = eval_dir / f"gate_{stage}.json"
    if not gate.is_file():
        return False
    with open(gate, encoding="utf-8") as f:
        return bool(json.load(f).get("passed"))


def training_complete(run_dir: Path, max_steps: int = 2000) -> bool:
    if final_exists(run_dir):
        return True
    pointer = load_latest_pointer(str(run_dir))
    if pointer and int(pointer.get("global_step", 0)) >= max_steps:
        return True
    return False


def consolidate_run2(log: QueueLogger) -> None:
    """Merge legacy split Run 2 dirs into one ordered checkpoint tree."""
    if final_exists(RUN2_OUT):
        return
    pointer = load_latest_pointer(str(RUN2_OUT))
    main_step, main_dir = find_latest_step_dir(str(RUN2_OUT))
    alt_step, alt_dir = find_latest_step_dir(str(RUN2_ALT))

    effective_main = int(pointer["global_step"]) if pointer else (main_step or 0)
    # Alt run started from step_500 weights; folder step_N means 500+N effective.
    effective_alt = (500 + alt_step) if alt_dir and alt_step else 0

    if alt_dir and effective_alt > effective_main:
        target_step = effective_alt
        target_dir = RUN2_OUT / f"step_{target_step:06d}"
        log.line(
            f"Consolidating Run 2 checkpoint: {alt_dir} -> {target_dir} "
            f"(effective global_step={target_step})"
        )
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(alt_dir, target_dir)
        meta = {
            "global_step": target_step,
            "micro_step": 0,
            "loss": None,
            "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "adapter_dir": str(target_dir).replace("\\", "/"),
            "projector_weights": str(target_dir / "projector_weights.pth").replace("\\", "/"),
            "note": "consolidated from phase_c_run2_chatml_resume_from_500",
        }
        with open(target_dir / "checkpoint_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        save_latest_pointer(
            str(RUN2_OUT),
            global_step=target_step,
            step_dir=str(target_dir),
            micro_step=0,
            max_global_steps=2000,
            has_training_state=False,
            init_checkpoint=str(RUN1_OUT / "final"),
        )
        return

    if effective_main > 0 and not pointer:
        step_dir = main_dir or str(RUN2_OUT / f"step_{effective_main:06d}")
        if verify_step_dir(step_dir) == []:
            save_latest_pointer(
                str(RUN2_OUT),
                global_step=effective_main,
                step_dir=step_dir,
                micro_step=0,
                max_global_steps=2000,
                has_training_state=False,
                init_checkpoint=str(RUN1_OUT / "final"),
            )
            log.line(f"Wrote latest.json for Run 2 at global_step={effective_main}")


def train_run(
    log: QueueLogger,
    *,
    title: str,
    output_dir: Path,
    init_checkpoint: Path,
    learning_rate: float,
    use_chatml: bool,
) -> int:
    stale = stale_lock_message()
    if stale and "held by pid" in stale:
        log.line(f"SKIP: {stale}")
        return 1
    if stale:
        log.line(stale)
        clear_stale_lock()

    cmd = [
        sys.executable,
        str(ROOT / "train_phase_c.py"),
        "--subset_size",
        "30000",
        "--learning_rate",
        str(learning_rate),
        "--output_dir",
        str(output_dir),
        "--init_checkpoint",
        str(init_checkpoint),
        "--max_global_steps",
        "2000",
        "--checkpoint_every",
        "250",
        "--auto_resume",
    ]
    if use_chatml:
        cmd.append("--use_chatml")
    return run_cmd(log, title, cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="KMVision week queue orchestrator")
    parser.add_argument(
        "--from-run",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Start at run N (default 1). Earlier completed stages are skipped.",
    )
    args = parser.parse_args()

    log = QueueLogger(MAIN_LOG)
    log.line(f"Week queue start (from-run={args.from_run})")

    if args.from_run <= 1:
        rc = run_cmd(log, "compress gate", [sys.executable, str(ROOT / "scripts" / "check_compress_gate.py")])
        if rc != 0:
            rc = run_cmd(
                log,
                "compress labels",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compress_labels.py"),
                    "--input-dir",
                    str(DATASET_ROOT / "train_1" / "labels"),
                    "--output-dir",
                    str(DATASET_ROOT / "labels_compressed"),
                ],
            )
            if rc != 0:
                return rc
            rc = run_cmd(log, "compress gate", [sys.executable, str(ROOT / "scripts" / "check_compress_gate.py")])
            if rc != 0:
                return rc

        if not final_exists(RUN1_OUT):
            if verify_step_dir(str(PHASE_B)) != []:
                log.line(f"FATAL: missing Phase B init at {PHASE_B}")
                return 1
            rc = train_run(
                log,
                title="RUN 1 train (minified)",
                output_dir=RUN1_OUT,
                init_checkpoint=PHASE_B,
                learning_rate=5e-5,
                use_chatml=False,
            )
            if rc != 0:
                return rc
        else:
            log.line("SKIP: Run 1 training complete (final exists)")

        rc = run_cmd(log, "verify Run 1 final", [sys.executable, str(ROOT / "scripts" / "verify_checkpoint.py"), str(RUN1_OUT / "final")])
        if rc != 0:
            return rc

        if not (EVAL1 / "latest_summary.json").is_file() or not gate_passed(EVAL1, "run1"):
            rc = run_cmd(
                log,
                "EVAL 1",
                [
                    sys.executable,
                    str(ROOT / "eval_inference.py"),
                    "--checkpoint",
                    str(RUN1_OUT / "final").replace("\\", "/"),
                    "--category",
                    "km",
                    "--max-samples",
                    "12",
                    "--output-dir",
                    str(EVAL1),
                ],
            )
            if rc != 0:
                return rc
            rc = run_cmd(
                log,
                "gate run1",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_eval_gate.py"),
                    "--stage",
                    "run1",
                    "--results-dir",
                    str(EVAL1),
                ],
            )
            if rc != 0:
                return rc
        else:
            log.line("SKIP: Run 1 eval + gate already passed")

    if args.from_run <= 2:
        consolidate_run2(log)

        if not training_complete(RUN2_OUT):
            init_ckpt = RUN1_OUT / "final"
            pointer = load_latest_pointer(str(RUN2_OUT))
            if pointer and Path(pointer["step_dir"]).is_dir():
                log.line(f"Run 2 will auto-resume from global_step={pointer.get('global_step')}")
            rc = train_run(
                log,
                title="RUN 2 train (ChatML)",
                output_dir=RUN2_OUT,
                init_checkpoint=init_ckpt,
                learning_rate=5e-5,
                use_chatml=True,
            )
            if rc != 0:
                return rc
        else:
            log.line("SKIP: Run 2 training complete")

        rc = run_cmd(log, "verify Run 2 final", [sys.executable, str(ROOT / "scripts" / "verify_checkpoint.py"), str(RUN2_OUT / "final")])
        if rc != 0:
            return rc

        if not gate_passed(EVAL2, "run2"):
            rc = run_cmd(
                log,
                "EVAL 2",
                [
                    sys.executable,
                    str(ROOT / "eval_inference.py"),
                    "--checkpoint",
                    str(RUN2_OUT / "final").replace("\\", "/"),
                    "--category",
                    "km",
                    "--max-samples",
                    "12",
                    "--output-dir",
                    str(EVAL2),
                ],
            )
            if rc != 0:
                return rc
            rc = run_cmd(
                log,
                "gate run2",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_eval_gate.py"),
                    "--stage",
                    "run2",
                    "--results-dir",
                    str(EVAL2),
                    "--previous-summary",
                    str(EVAL1 / "latest_summary.json"),
                ],
            )
            if rc != 0:
                return rc
        else:
            log.line("SKIP: Run 2 eval + gate already passed")

    if args.from_run <= 3:
        if not training_complete(RUN3_OUT):
            rc = train_run(
                log,
                title="RUN 3 train (ChatML low LR)",
                output_dir=RUN3_OUT,
                init_checkpoint=RUN2_OUT / "final",
                learning_rate=1e-5,
                use_chatml=True,
            )
            if rc != 0:
                return rc
        else:
            log.line("SKIP: Run 3 training complete")

        rc = run_cmd(log, "verify Run 3 final", [sys.executable, str(ROOT / "scripts" / "verify_checkpoint.py"), str(RUN3_OUT / "final")])
        if rc != 0:
            return rc

        if not gate_passed(EVAL3, "run3"):
            rc = run_cmd(
                log,
                "EVAL 3",
                [
                    sys.executable,
                    str(ROOT / "eval_inference.py"),
                    "--checkpoint",
                    str(RUN3_OUT / "final").replace("\\", "/"),
                    "--category",
                    "km",
                    "--max-samples",
                    "12",
                    "--output-dir",
                    str(EVAL3),
                ],
            )
            if rc != 0:
                return rc
            rc = run_cmd(
                log,
                "gate run3",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_eval_gate.py"),
                    "--stage",
                    "run3",
                    "--results-dir",
                    str(EVAL3),
                    "--previous-summary",
                    str(EVAL2 / "latest_summary.json"),
                ],
            )
            if rc != 0:
                return rc
        else:
            log.line("SKIP: Run 3 eval + gate already passed")

    log.line("Week queue completed OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
