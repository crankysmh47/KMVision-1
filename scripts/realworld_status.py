"""Report real-world data collection and labeling progress."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real_dataset"))

from config import TARGETS, curated_count, inbox_count, labels_dir, list_pngs, unlabeled_queue  # noqa: E402


def main() -> int:
    report = {"targets": TARGETS, "types": {}}
    for chart_type in TARGETS:
        label_files = list(labels_dir(chart_type).glob("*.json")) if labels_dir(chart_type).exists() else []
        report["types"][chart_type] = {
            "inbox_images": inbox_count(chart_type),
            "curated_images": curated_count(chart_type),
            "labeled": len(label_files),
            "unlabeled_queue": len(unlabeled_queue(chart_type)),
            "target": TARGETS[chart_type],
            "pct_of_target": round(curated_count(chart_type) / TARGETS[chart_type] * 100, 1),
        }

    out = ROOT / "real_dataset" / "status_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
