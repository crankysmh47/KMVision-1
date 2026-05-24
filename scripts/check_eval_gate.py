"""
Pass/fail gate on eval summary JSON for automated week queue.

Usage:
  python scripts/check_eval_gate.py --stage run1 --results-dir evaluation/results/run1_minified
  python scripts/check_eval_gate.py --stage run2 --results-dir evaluation/results/run2_chatml ^
      --previous-summary evaluation/results/run1_minified/latest_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GATES = os.path.join(ROOT, "config", "eval_gates.json")


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_summary(results_dir: str) -> str:
    latest = os.path.join(results_dir, "latest_summary.json")
    if os.path.isfile(latest):
        return latest
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results dir not found: {results_dir}")
    candidates = [
        os.path.join(results_dir, name)
        for name in os.listdir(results_dir)
        if name.endswith("_summary.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No summary JSON in {results_dir}")
    return max(candidates, key=os.path.getmtime)


def check_absolute(summary: dict, minimums: dict) -> list[str]:
    failures = []
    for key, floor in minimums.items():
        val = summary.get(key)
        if val is None:
            failures.append(f"{key}: missing (required >= {floor})")
        elif val < floor:
            failures.append(f"{key}: {val:.4f} < {floor}")
    return failures


def check_baseline_delta(summary: dict, baseline: dict, min_delta: dict) -> list[str]:
    failures = []
    for key, delta_floor in min_delta.items():
        cur = summary.get(key)
        base = baseline.get(key)
        if cur is None or base is None:
            continue
        delta = cur - base
        if delta < delta_floor:
            failures.append(
                f"{key}: delta {delta:+.4f} vs baseline (need >= {delta_floor:+.4f}); "
                f"current={cur:.4f} baseline={base:.4f}"
            )
    return failures


def check_previous_relative(
    summary: dict,
    previous: dict,
    metrics: list[str],
    min_relative: float,
) -> list[str]:
    failures = []
    for key in metrics:
        cur = summary.get(key)
        prev = previous.get(key)
        if cur is None or prev is None or prev == 0:
            continue
        ratio = cur / prev
        if ratio < min_relative:
            failures.append(
                f"{key}: ratio {ratio:.4f} vs previous (need >= {min_relative}); "
                f"current={cur:.4f} previous={prev:.4f}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval metric gate for week queue.")
    parser.add_argument("--stage", required=True, choices=["run1", "run2", "run3"])
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--summary", default=None, help="Override summary JSON path.")
    parser.add_argument("--previous-summary", default=None)
    parser.add_argument("--gates", default=DEFAULT_GATES)
    parser.add_argument("--warn-only", action="store_true", help="Print failures but exit 0.")
    args = parser.parse_args()

    gates_cfg = load_json(args.gates)
    stage_cfg = gates_cfg["runs"][args.stage]
    summary_path = args.summary or find_summary(args.results_dir)
    summary = load_json(summary_path)

    print(f"Gate check: stage={args.stage}")
    print(f"  summary: {summary_path}")
    failures: list[str] = []

    failures.extend(check_absolute(summary, stage_cfg.get("absolute_minimums", {})))

    if args.stage == "run1":
        baseline_path = os.path.join(ROOT, gates_cfg.get("baseline_summary", ""))
        if os.path.isfile(baseline_path):
            baseline = load_json(baseline_path)
            failures.extend(
                check_baseline_delta(
                    summary, baseline, stage_cfg.get("vs_baseline_min_delta", {})
                )
            )
        else:
            print(f"  WARN: baseline not found at {baseline_path}, skipping delta check")

    if args.stage in ("run2", "run3") and args.previous_summary:
        if os.path.isfile(args.previous_summary):
            previous = load_json(args.previous_summary)
            failures.extend(
                check_previous_relative(
                    summary,
                    previous,
                    stage_cfg.get("vs_previous_metrics", ["mean_overall_score"]),
                    stage_cfg.get("vs_previous_min_relative", 0.9),
                )
            )
        else:
            failures.append(f"previous summary missing: {args.previous_summary}")

    report_path = os.path.join(args.results_dir, f"gate_{args.stage}.json")
    os.makedirs(args.results_dir, exist_ok=True)
    report = {
        "stage": args.stage,
        "passed": len(failures) == 0,
        "summary_path": summary_path,
        "failures": failures,
        "metrics": {k: summary.get(k) for k in stage_cfg.get("absolute_minimums", {})},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  report:  {report_path}")

    if failures:
        print("GATE FAILED:")
        for line in failures:
            print(f"  - {line}")
        return 0 if args.warn_only else 1

    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
