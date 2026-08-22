"""
Stitch Stage 2 tile predictions back into per-arm clinical KM coordinates.

Takes normalized flat_xy tile outputs + `_meta` from tile labels, inverts to
clinical (time, survival), deduplicates overlap between adjacent tiles, and
emits verbose KM arm payloads suitable for evaluation/metrics.py.

Stitch semantics (post Phase-0 repair, 2026-08-22)
----------------------------------------------------
- Tile predictions are parsed unconditionally: a record carries either a
  pre-parsed dict (`prediction` / `parsed`) or raw text (`prediction_raw`)
  which is parsed via `evaluation.parse_output.extract_stage2_json`
  (repair heuristics included). This used to happen only when a record had
  no `_meta`, which meant every eval-JSONL record (all of which carry
  `_meta`) was stitched with EMPTY coordinates. That bug produced the
  frozen 0.590 E2E number, which measured no model output at all and is
  retired.
- `strict=True` (default): a tile prediction that is missing, unparseable,
  out of bounds, or structurally invalid RAISES `StitchError`. There is no
  silent fallback of any kind. Evaluation must fail loudly.
- `strict=False` (lenient): unusable tiles are skipped and counted; they
  are never substituted with anything else.
- Every stitched arm records provenance (`_meta.stitch`): per-tile
  prediction_source, point/censor counts, and the stitch version, so an
  E2E score can always be traced to the predictions that produced it.

Usage:
  python scripts/stitch_tiles.py --jsonl eval.jsonl [--lenient]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.parse_output import extract_stage2_json
from scripts.stage2_coordinate_transform import normalized_local_to_clinical
from stage2_common import coords_to_pairs

STITCH_VERSION = "stitch_v2_provenance_strict"
TIME_DEDUPE_TOL = 0.25  # clinical months — overlap between adjacent tiles
CENSOR_DEDUPE_TOL = 0.25


class StitchError(ValueError):
    """Loud failure for unusable tile predictions (strict mode)."""


# --------------------------------------------------------------------------
# Prediction parsing + provenance
# --------------------------------------------------------------------------

def parse_tile_prediction(rec: dict) -> Tuple[Optional[dict], str]:
    """
    Extract a usable Stage 2 prediction dict from a tile record.

    Resolution order:
      1. rec["prediction"] / rec["parsed"]  (already-parsed dict)
      2. rec["prediction_raw"] parsed via extract_stage2_json (with repair)

    Returns (pred_dict_or_None, source) where source is one of:
      "stage2_tile"       — prediction came with the record (parsed dict)
      "stage2_tile_raw"   — prediction parsed from prediction_raw
    """
    pred = rec.get("prediction") or rec.get("parsed")
    if isinstance(pred, dict) and (
        "points" in pred or "censors" in pred or "arm_id" in pred
    ):
        return pred, "stage2_tile"
    raw = rec.get("prediction_raw")
    if isinstance(raw, str) and raw.strip():
        parsed, _err = extract_stage2_json(raw)
        if isinstance(parsed, dict):
            return parsed, "stage2_tile_raw"
    return None, "missing"


def _assert_normalized_pairs(pairs: Sequence[Sequence[float]], kind: str, tile_id: str) -> None:
    for pair in pairs:
        if len(pair) < 2:
            raise StitchError(f"{tile_id}: malformed {kind} pair {pair!r}")
        x, y = pair[0], pair[1]
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise StitchError(
                f"{tile_id}: {kind} normalized coordinate out of [0,1]: ({x}, {y})"
            )


def _assert_clinical_points(points: Sequence[Sequence[float]], tile_id: str) -> None:
    for t, s in points:
        if float(t) < 0.0:
            raise StitchError(f"{tile_id}: negative clinical time {t} after inverse transform")
        if not (0.0 <= float(s) <= 1.0):
            raise StitchError(f"{tile_id}: survival {s} out of [0,1] after inverse transform")


# --------------------------------------------------------------------------
# Coordinate transforms
# --------------------------------------------------------------------------

def tile_pred_to_clinical(
    pred_obj: dict,
    meta: dict,
    *,
    tile_id: str = "<unknown tile>",
    check_bounds: bool = True,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Convert flat/nested tile prediction + _meta to clinical points and censors.

    Censor events travel through the SAME inverse transform as curve points
    and remain in their own dedicated list — their semantic type is never
    merged into anonymous curve coordinates.
    """
    if meta.get("coordinate_space") != "normalized_local":
        points = coords_to_pairs(pred_obj.get("points", []))
        censors = coords_to_pairs(pred_obj.get("censors", []))
        return points, censors

    tile_origin = meta["tile_origin"]
    plot_bbox = meta["plot_bbox"]
    axis_max = meta["axis_max"]

    point_pairs = coords_to_pairs(pred_obj.get("points", []))
    censor_pairs = coords_to_pairs(pred_obj.get("censors", []))
    if check_bounds:
        _assert_normalized_pairs(point_pairs, "point", tile_id)
        _assert_normalized_pairs(censor_pairs, "censor", tile_id)

    points: List[List[float]] = []
    for x_norm, y_norm in point_pairs:
        t, s = normalized_local_to_clinical(
            x_norm, y_norm,
            tile_origin=tile_origin, plot_bbox=plot_bbox, axis_max=axis_max,
        )
        points.append([t, s])

    censors: List[List[float]] = []
    for x_norm, y_norm in censor_pairs:
        t, s = normalized_local_to_clinical(
            x_norm, y_norm,
            tile_origin=tile_origin, plot_bbox=plot_bbox, axis_max=axis_max,
        )
        censors.append([t, s])

    if check_bounds:
        _assert_clinical_points(points, tile_id)
        _assert_clinical_points(censors, tile_id)

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


# --------------------------------------------------------------------------
# Stitching
# --------------------------------------------------------------------------

def stitch_arm_tiles(
    tile_records: Sequence[dict],
    *,
    strict: bool = True,
) -> Tuple[List[List[float]], List[float], List[dict]]:
    """
    Merge multiple tile predictions for one arm on one chart.

    Each record: {"prediction": {...}, "meta": {...}} or eval JSONL row with
    a label path (meta then comes from the tile label's `_meta`).

    Returns (points, censor_times, provenance_rows). Raises StitchError in
    strict mode when any tile prediction is missing/unparseable/out of bounds.
    """
    all_points: List[List[float]] = []
    all_censors: List[List[float]] = []
    provenance: List[dict] = []

    for rec in tile_records:
        tile_id = Path(str(rec.get("label") or rec.get("image") or "<unknown>")).stem
        pred, source = parse_tile_prediction(rec)

        if pred is None:
            if strict:
                raise StitchError(
                    f"Tile {tile_id}: prediction missing or unparseable "
                    f"(prediction_raw={(rec.get('prediction_raw') or '')[:80]!r}). "
                    f"Refusing to continue without a real prediction."
                )
            provenance.append(
                {"tile": tile_id, "prediction_source": "skipped_unparseable",
                 "points_in": 0, "censors_in": 0}
            )
            continue

        points_in = len(coords_to_pairs(pred.get("points", [])))
        censors_in = len(coords_to_pairs(pred.get("censors", [])))
        if points_in == 0 and censors_in == 0:
            if strict:
                raise StitchError(f"Tile {tile_id}: parsed prediction contains no points or censors")
            provenance.append(
                {"tile": tile_id, "prediction_source": source,
                 "points_in": 0, "censors_in": 0, "note": "empty prediction"}
            )
            continue

        meta = rec.get("meta") or rec.get("_meta") or {}
        if not meta and rec.get("label"):
            with open(rec["label"], encoding="utf-8") as f:
                meta = json.load(f).get("_meta", {})
        if not meta:
            if strict:
                raise StitchError(f"Tile {tile_id}: no _meta for inverse transform")
            provenance.append(
                {"tile": tile_id, "prediction_source": "skipped_no_meta",
                 "points_in": points_in, "censors_in": censors_in}
            )
            continue

        pts, cens = tile_pred_to_clinical(pred, meta, tile_id=tile_id, check_bounds=strict)
        all_points.extend(pts)
        all_censors.extend(cens)
        provenance.append(
            {"tile": tile_id, "prediction_source": source,
             "points_in": points_in, "censors_in": censors_in}
        )

    points = _dedupe_points_by_time(all_points)
    censor_times = _dedupe_censors_by_time(all_censors)
    return points, censor_times, provenance


def stitch_chart_from_tiles(
    tile_records: Sequence[dict],
    *,
    chart_gt: Optional[dict] = None,
    strict: bool = True,
) -> dict:
    """
    Build verbose KM chart JSON from stitched tile groups.

    Uses GT chart skeleton (axes, arm labels) when provided; otherwise minimal shell.
    strict=True raises StitchError on any unusable tile prediction (default for eval).
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

    arm_stitch_meta: List[dict] = []
    skipped_tiles = 0
    for arm_id, group in sorted(by_arm.items()):
        points, censor_times, provenance = stitch_arm_tiles(group, strict=strict)
        skipped_tiles += sum(
            1 for row in provenance if str(row.get("prediction_source", "")).startswith("skipped")
        )
        out["arms"].append(
            {
                "treatment_label": arm_id,
                "coordinates": points,
                "censoring_ticks": censor_times,
            }
        )
        arm_stitch_meta.append({"arm_id": arm_id, "tiles": provenance})

    out["_meta"] = {
        "source_chart": source_chart,
        "stitched_from_tiles": len(tile_records),
        "stitch": {
            "version": STITCH_VERSION,
            "strict": bool(strict),
            "skipped_tiles": skipped_tiles,
            "arms": arm_stitch_meta,
        },
    }
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
    p.add_argument(
        "--lenient", action="store_true",
        help="Skip unusable tile predictions instead of raising (not recommended for evaluation).",
    )
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
        stitched[chart_stem] = stitch_chart_from_tiles(
            records, chart_gt=chart_gt, strict=not args.lenient
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stitched, f, indent=2)

    total_points = sum(len(c) for chart in stitched.values() for c in
                       (arm["coordinates"] for arm in chart["arms"]))
    total_censors = sum(len(arm["censoring_ticks"]) for chart in stitched.values() for arm in chart["arms"])
    print(f"Stitched {len(stitched)} charts from {len(groups)} source charts -> {out_path}")
    print(f"Stitched coordinate totals: {total_points} points, {total_censors} censors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
