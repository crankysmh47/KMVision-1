"""Archive Phase B checkpoint, eval artifacts, and write baseline report."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "archives", "phase_b_baseline")


def copytree(src: str, dst: str) -> None:
    if not os.path.isdir(src):
        print(f"SKIP missing: {src}")
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied {src} -> {dst}")


def copyfile(src: str, dst: str) -> None:
    if not os.path.isfile(src):
        print(f"SKIP missing: {src}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    os.makedirs(ARCHIVE, exist_ok=True)

    copytree(
        os.path.join(ROOT, "checkpoints", "phase_b", "final"),
        os.path.join(ARCHIVE, "checkpoints", "phase_b_final"),
    )
    copytree(
        os.path.join(ROOT, "evaluation", "results"),
        os.path.join(ARCHIVE, "evaluation_results"),
    )
    copyfile(
        r"C:\sem4\KMVision-1 Data\dataset\split_manifest.json",
        os.path.join(ARCHIVE, "split_manifest.json"),
    )

    km_summary = os.path.join(
        ROOT, "evaluation", "results", "km_final", "eval_20260523T220957Z_summary.json"
    )
    summary_data = {}
    if os.path.isfile(km_summary):
        with open(km_summary, encoding="utf-8") as f:
            summary_data = json.load(f)

    report = f"""# Phase B Baseline Evaluation Report

Archived: {stamp}

## Checkpoint
- `checkpoints/phase_b/final` (LoRA adapter + projector)
- Copied to `archives/phase_b_baseline/checkpoints/phase_b_final/`

## Dataset split
- Train: 100,000 samples in `train_1/` (Phase B selection logic, seed=42)
- Test: 399,972 samples in `testing/`
- Manifest: `split_manifest.json`

## KM holdout eval (12 charts, checkpoint final, max_new_tokens=2048)

| Metric | Value |
|--------|-------|
| Overall score | {summary_data.get('mean_overall_score', 'N/A')} |
| JSON valid (after repair) | {summary_data.get('json_valid_rate', 'N/A')} |
| Chart type accuracy | {summary_data.get('chart_type_accuracy', 'N/A')} |
| Text score | {summary_data.get('mean_text_score', 'N/A')} |
| Structure score | {summary_data.get('mean_structure_score', 'N/A')} |
| Numeric score | {summary_data.get('mean_numeric_score', 'N/A')} |
| Censoring score | {summary_data.get('mean_censoring_score', 'N/A')} |
| Curve RMSE | {summary_data.get('mean_coordinate_rmse', 'N/A')} |

## Diagnosis
Phase B used `max_length=768` with verbose JSON labels. Long KM charts were truncated during
training, causing missing arms, zero censoring recall, and mid-JSON generation at inference.

## Phase C plan
Minified JSON + step-aware coordinate subsampling; continue from Phase B final; no ChatML yet.

## Artifacts
- `archives/phase_b_baseline/evaluation_results/` — all JSONL + summaries
"""
    report_path = os.path.join(ARCHIVE, "EVAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
