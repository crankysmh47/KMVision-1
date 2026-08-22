"""
Stitch Stage 2 tile predictions back into per-arm clinical KM coordinates.

Takes normalized flat_xy tile outputs + `_meta` from tile labels, inverts to
clinical (time, survival), deduplicates overlap between adjacent tiles, and
emits verbose KM arm payloads suitable for evaluation/metrics.py.

Usage:
  python scripts/stitch_tiles.py --tile-labels-dir "path/to/labels/km" --predictions predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.parse_output import extract_stage2_json
from scripts.stage2_coordinate_transform import normalized_local_to_clinical
from stage2_common import coords_to_pairs

TIME_DEDUPE_TOL = 0.25  # clinical months — overlap between adjacent tiles
CENSOR_DEDUPE_TOL = 0.25


def tile_pred_to_clinical(pred_obj: dict, meta: dict) -> Tuple[List[List[float]], List[List[float]]]:
    """Convert flat/nested tile prediction + _meta to clinical points and censors."""
    if meta.get("coordinate_space") != "normalized_local":
        points = coords_to_pairs(pred_obj.get("points", []))
        censors = coords_to_pairs(pred_obj.get("censors", []))
        return points, censors

    tile_origin = meta["tile_origin"]
    plot_bbox = meta["plot_bbox"]
    axis_max = meta["axis_max"]

    points: List[List[float]] = []
    for x_norm, y_norm in coords_to_pairs(pred_obj.get("points", [])):
        t, s = normalized_local_to_clinical(
            x_norm,
            y_norm,
            tile_origin=tile_origin,
            plot_bbox=plot_bbox,
            axis_max=axis_max,
        )
        points.append([t, s])

    censors: List[List[float]] = []
    for x_norm, y_norm in coords_to_pairs(pred_obj.get("censors", [])):
        t, s = normalized_local_to_clinical(
            x_norm,
            y_norm,
            tile_origin=tile_origin,
            plot_bbox=plot_bbox,
            axis_max=axis_max,
        )
        censors.append([t, s])

    return points, censors


def _dedupe_points_by_time(
    points: Sequence[Sequence[float]],
    *,
    tol: float = TIME_DEDUPE_TOL,
) -> List[List[float]]:
    """Keep earliest occurrence per time bucket (sorted by time)."""
    if not points:
        return []
    sorted_pts = sorted((float(p[0]), float(p[1])) for p in points if len(p) >= 2)
    out: List[List[float]] = []
    for t, s in sorted_pts:
        if out and abs(out[-1][0] - t) <= tol:
            continue
        out.append([round(t, 4), round(max(0.0, min(1.0, s)), 6)])
    return out


def _dedupe_censors_by_time(
    censors: Sequence[Sequence[float]],
    *,
    tol: float = CENSOR_DEDUPE_TOL,
) -> List[float]:
    times = sorted(float(c[0]) for c in censors if len(c) >= 1)
    out: List[float] = []
    for t in times:
        if out and abs(out[-1] - t) <= tol:
            continue
        out.append(round(t, 4))
    return out


def stitch_arm_tiles(
    tile_records: Sequence[dict],
) -> Tuple[List[List[float]], List[float]]:
    """
    Merge multiple tile predictions for one arm on one chart.

    Each record: {"prediction": {...}, "meta": {...}} or eval JSONL row with label path.
    """
    all_points: List[List[float]] = []
    all_censors: List[List[float]] = []

    for rec in tile_records:
        pred = rec.get("prediction") or rec.get("parsed") or {}
        meta = rec.get("meta") or rec.get("_meta") or {}
        if not meta and rec.get("label"):
            with open(rec["label"], encoding="utf-8") as f:
                label_obj = json.load(f)
            meta = label_obj.get("_meta", {})
            if not pred:
                raw = rec.get("prediction_raw", "")
                parsed, _ = extract_stage2_json(raw)
                pred = parsed or {}

        pts, cens = tile_pred_to_clinical(pred or {}, meta)
        all_points.extend(pts)
        all_censors.extend(cens)

    points = _dedupe_points_by_time(all_points)
    censor_times = _dedupe_censors_by_time(all_censors)
    return points, censor_times


def stitch_chart_from_tiles(
    tile_records: Sequence[dict],
    *,
    chart_gt: Optional[dict] = None,
) -> dict:
    """
    Build verbose KM chart JSON from stitched tile groups.

    Uses GT chart skeleton (axes, arm labels) when provided; otherwise minimal shell.
    """
    by_arm: DefaultDict[str, List[dict]] = defaultdict(list)
    source_chart = None

    for rec in tile_records:
        label_path = rec.get("label")
        label_obj: dict = {}
        if label_path:
            with open(label_path, encoding="utf-8") as f:
                label_obj = json.load(f)
        meta = rec.get("meta") or rec.get("_meta") or label_obj.get("_meta", {})
        arm_id = str(rec.get("arm_id") or label_obj.get("arm_id") or "unknown")
        if meta:
            source_chart = meta.get("source_chart", source_chart)
        by_arm[arm_id].append({**rec, "meta": meta, "arm_id": arm_id})

    if chart_gt:
        out = {
            "chart_type": chart_gt.get("chart_type", "kaplan_meier"),
            "axes": chart_gt.get("axes", {}),
            "arms": [],
        }
    else:
        out = {
            "chart_type": "kaplan_meier",
            "axes": {"x": {"label": "", "max_value": 1.0}, "y": {"label": "", "max_value": 1.0}},
            "arms": [],
        }

    for arm_id, group in sorted(by_arm.items()):
        points, censor_times = stitch_arm_tiles(group)
        out["arms"].append(
            {
                "treatment_label": arm_id,
                "coordinates": points,
                "censoring_ticks": censor_times,
            }
        )

    if source_chart:
        out["_meta"] = {"source_chart": source_chart, "stitched_from_tiles": len(tile_records)}
    return out


def group_eval_jsonl_by_chart(jsonl_path: Path) -> Dict[str, List[dict]]:
    """Group Stage 2 eval JSONL rows by source chart stem from tile label _meta."""
    groups: DefaultDict[str, List[dict]] = defaultdict(list)
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        with open(rec["label"], encoding="utf-8") as f:
            label_obj = json.load(f)
        chart_stem = label_obj.get("_meta", {}).get("source_chart", "unknown")
        rec["_meta"] = label_obj.get("_meta", {})
        rec["arm_id"] = label_obj.get("arm_id", rec.get("arm_id", "unknown"))
        groups[chart_stem].append(rec)
    return dict(groups)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stitch Stage 2 tile preds -> clinical KM JSON.")
    p.add_argument("--jsonl", type=str, required=True, help="Stage 2 eval JSONL with prediction_raw")
    p.add_argument("--chart-gt-dir", type=str, default=None, help="Verbose KM labels for chart skeleton")
    p.add_argument("--output", type=str, default="evaluation/results/stitched_charts.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        print(f"Missing JSONL: {jsonl_path}")
        return 1

    groups = group_eval_jsonl_by_chart(jsonl_path)
    gt_dir = Path(args.chart_gt_dir) if args.chart_gt_dir else None

    stitched: Dict[str, dict] = {}
    for chart_stem, records in groups.items():
        chart_gt = None
        if gt_dir:
            gt_path = gt_dir / f"{chart_stem}.json"
            if gt_path.is_file():
                with open(gt_path, encoding="utf-8") as f:
                    chart_gt = json.load(f)
        stitched[chart_stem] = stitch_chart_from_tiles(records, chart_gt=chart_gt)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stitched, f, indent=2)

    print(f"Stitched {len(stitched)} charts from {len(groups)} source charts -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
