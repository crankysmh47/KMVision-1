"""
Chart extraction evaluation metrics.

Scores model output against ground-truth JSON using structure-aware,
tolerance-based comparisons — not byte-level or per-digit alignment.

Primary focus: Kaplan-Meier charts; other chart types use a lighter generic scorer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from evaluation.parse_output import extract_json_from_text

JsonLike = Union[dict, str]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ChartScore:
    """Per-sample evaluation breakdown. overall_score is in [0, 1]."""

    overall_score: float
    json_valid: bool
    chart_type_match: Optional[bool]
    parse_error: Optional[str] = None

    # Sub-scores (each in [0, 1] when applicable)
    text_score: float = 0.0
    structure_score: float = 0.0
    numeric_score: float = 0.0
    censoring_score: float = 0.0

    # Diagnostic counts / errors
    fields_correct: int = 0
    fields_total: int = 0
    arms_matched: int = 0
    arms_ground_truth: int = 0
    arms_predicted: int = 0
    coordinate_rmse: Optional[float] = None
    coordinate_mae: Optional[float] = None
    axis_max_relative_error: Optional[float] = None

    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_extraction(
    ground_truth: JsonLike,
    prediction: JsonLike,
    *,
    chart_type: Optional[str] = None,
) -> ChartScore:
    """
    Score how much information the model extracted correctly.

    Args:
        ground_truth: Ground-truth dict or JSON string.
        prediction: Model output dict or raw text (JSON will be extracted).
        chart_type: Force chart type ('kaplan_meier', etc.). Inferred from GT if None.

    Returns:
        ChartScore with overall_score in [0, 1] and detailed breakdown.
    """
    gt = _ensure_dict(ground_truth)
    if gt is None:
        raise ValueError("ground_truth must be a dict or valid JSON string")

    if isinstance(prediction, str):
        pred, parse_err = extract_json_from_text(prediction)
    else:
        pred = prediction if isinstance(prediction, dict) else None
        parse_err = None if pred is not None else "prediction is not a dict"

    resolved_type = chart_type or gt.get("chart_type", "unknown")

    if pred is None:
        return ChartScore(
            overall_score=0.0,
            json_valid=False,
            chart_type_match=False,
            parse_error=parse_err,
            details={"chart_type": resolved_type},
        )

    type_match = _normalize_chart_type(pred.get("chart_type")) == _normalize_chart_type(
        gt.get("chart_type")
    )

    if resolved_type == "kaplan_meier" or gt.get("chart_type") == "kaplan_meier":
        return _score_km(gt, pred, type_match, parse_err)

    return _score_generic(gt, pred, type_match, parse_err)


def aggregate_scores(scores: Sequence[ChartScore]) -> dict:
    """Summarize a list of ChartScore results."""
    if not scores:
        return {"count": 0}

    n = len(scores)
    valid = [s for s in scores if s.json_valid]

    def mean_attr(attr: str, subset: Sequence[ChartScore]) -> float:
        if not subset:
            return 0.0
        return sum(getattr(s, attr) for s in subset) / len(subset)

    return {
        "count": n,
        "json_valid_rate": sum(1 for s in scores if s.json_valid) / n,
        "chart_type_accuracy": sum(1 for s in scores if s.chart_type_match) / n,
        "mean_overall_score": mean_attr("overall_score", scores),
        "mean_overall_score_valid_json": mean_attr("overall_score", valid),
        "mean_text_score": mean_attr("text_score", valid),
        "mean_structure_score": mean_attr("structure_score", valid),
        "mean_numeric_score": mean_attr("numeric_score", valid),
        "mean_censoring_score": mean_attr("censoring_score", valid),
        "mean_coordinate_rmse": _nanmean([s.coordinate_rmse for s in valid if s.coordinate_rmse is not None]),
        "mean_fields_correct_ratio": _nanmean(
            [s.fields_correct / s.fields_total for s in valid if s.fields_total > 0]
        ),
    }


# ---------------------------------------------------------------------------
# Kaplan-Meier scoring
# ---------------------------------------------------------------------------

# Composite weights (sum to 1.0)
_KM_WEIGHTS = {
    "json_valid": 0.10,
    "chart_type": 0.05,
    "text": 0.20,
    "structure": 0.15,
    "numeric": 0.40,
    "censoring": 0.10,
}

_TIME_ABS_TOL = 0.5
_TIME_REL_TOL = 0.02
_SURV_ABS_TOL = 0.05
_CENSOR_TIME_TOL = 1.0


def _score_km(
    gt: dict,
    pred: dict,
    type_match: bool,
    parse_err: Optional[str],
) -> ChartScore:
    text_score, text_details = _km_text_score(gt, pred)
    structure_score, struct_details, arms_matched = _km_structure_score(gt, pred)
    numeric_score, num_details, coord_rmse, coord_mae, axis_err, fields_ok, fields_total = (
        _km_numeric_score(gt, pred, struct_details.get("arm_pairs", []))
    )
    censoring_score, cens_details = _km_censoring_score(
        gt, pred, struct_details.get("arm_pairs", [])
    )

    fields_correct = fields_ok + text_details.get("text_fields_correct", 0)
    fields_total = fields_total + text_details.get("text_fields_total", 0)

    overall = (
        _KM_WEIGHTS["json_valid"] * 1.0
        + _KM_WEIGHTS["chart_type"] * (1.0 if type_match else 0.0)
        + _KM_WEIGHTS["text"] * text_score
        + _KM_WEIGHTS["structure"] * structure_score
        + _KM_WEIGHTS["numeric"] * numeric_score
        + _KM_WEIGHTS["censoring"] * censoring_score
    )

    return ChartScore(
        overall_score=round(overall, 4),
        json_valid=True,
        chart_type_match=type_match,
        parse_error=parse_err,
        text_score=round(text_score, 4),
        structure_score=round(structure_score, 4),
        numeric_score=round(numeric_score, 4),
        censoring_score=round(censoring_score, 4),
        fields_correct=fields_correct,
        fields_total=fields_total,
        arms_matched=arms_matched,
        arms_ground_truth=len(gt.get("arms", [])),
        arms_predicted=len(pred.get("arms", [])),
        coordinate_rmse=coord_rmse,
        coordinate_mae=coord_mae,
        axis_max_relative_error=axis_err,
        details={
            "text": text_details,
            "structure": struct_details,
            "numeric": num_details,
            "censoring": cens_details,
        },
    )


def _km_text_score(gt: dict, pred: dict) -> Tuple[float, dict]:
    """Label / legend text similarity (OCR-tolerant)."""
    checks: List[float] = []
    correct = 0
    total = 0

    gt_axes = gt.get("axes", {})
    pred_axes = pred.get("axes", {})

    for axis_key in ("x", "y"):
        gt_label = _nested_label(gt_axes, axis_key)
        pred_label = _nested_label(pred_axes, axis_key)
        if gt_label is not None:
            total += 1
            sim = _text_similarity(gt_label, pred_label or "")
            checks.append(sim)
            if sim >= 0.85:
                correct += 1

    gt_arms = gt.get("arms", [])
    pred_arms = pred.get("arms", [])
    pairs = _match_arms_by_label(gt_arms, pred_arms)
    for gt_arm, pred_arm in pairs:
        if gt_arm is None:
            continue
        total += 1
        sim = _text_similarity(
            gt_arm.get("treatment_label", ""),
            (pred_arm or {}).get("treatment_label", ""),
        )
        checks.append(sim)
        if sim >= 0.85:
            correct += 1

    # Unmatched GT arms count as missed text fields
    for gt_arm in gt_arms:
        if not any(p[0] is gt_arm for p in pairs if p[0] is not None):
            total += 1
            checks.append(0.0)

    score = sum(checks) / len(checks) if checks else 0.0
    return score, {
        "text_fields_correct": correct,
        "text_fields_total": total,
        "per_field_similarity": checks,
    }


def _km_structure_score(gt: dict, pred: dict) -> Tuple[float, dict, int]:
    """Arm count and arm pairing quality."""
    gt_arms = gt.get("arms", [])
    pred_arms = pred.get("arms", [])
    if not gt_arms:
        return 0.0, {"arm_pairs": []}, 0

    pairs = _match_arms_by_label(gt_arms, pred_arms)
    matched = sum(1 for g, p in pairs if g is not None and p is not None)

    count_score = 1.0 - min(abs(len(gt_arms) - len(pred_arms)), len(gt_arms)) / len(gt_arms)
    match_score = matched / len(gt_arms)

    structure = 0.5 * count_score + 0.5 * match_score
    serializable_pairs = [
        {
            "gt_label": (g or {}).get("treatment_label"),
            "pred_label": (p or {}).get("treatment_label"),
            "label_similarity": _text_similarity(
                (g or {}).get("treatment_label", ""),
                (p or {}).get("treatment_label", ""),
            )
            if g and p
            else 0.0,
        }
        for g, p in pairs
    ]
    return structure, {"arm_pairs": pairs, "arm_pair_labels": serializable_pairs}, matched


def _km_numeric_score(
    gt: dict,
    pred: dict,
    arm_pairs: List[Tuple[Optional[dict], Optional[dict]]],
) -> Tuple[float, dict, Optional[float], Optional[float], Optional[float], int, int]:
    """Curve shape, axis max, and per-point accuracy."""
    scores: List[float] = []
    rmse_list: List[float] = []
    mae_list: List[float] = []
    points_correct = 0
    points_total = 0
    scalars_correct = 0
    scalars_total = 0

    # x-axis max_value
    gt_x_max = _nested_max(gt.get("axes", {}), "x")
    pred_x_max = _nested_max(pred.get("axes", {}), "x")
    axis_rel_err = None
    if gt_x_max is not None and gt_x_max > 0:
        scalars_total += 1
        if pred_x_max is not None:
            axis_rel_err = abs(pred_x_max - gt_x_max) / gt_x_max
            scores.append(_scalar_closeness(pred_x_max, gt_x_max, rel_tol=0.1, abs_tol=2.0))
            if axis_rel_err <= 0.1:
                scalars_correct += 1
        else:
            scores.append(0.0)

    # y max (usually 1.0)
    gt_y_max = _nested_max(gt.get("axes", {}), "y")
    pred_y_max = _nested_max(pred.get("axes", {}), "y")
    if gt_y_max is not None:
        scalars_total += 1
        if pred_y_max is not None:
            scores.append(_scalar_closeness(pred_y_max, gt_y_max, rel_tol=0.05, abs_tol=0.05))
            if abs(pred_y_max - gt_y_max) <= 0.05:
                scalars_correct += 1
        else:
            scores.append(0.0)

    curve_details = []
    for gt_arm, pred_arm in arm_pairs:
        if gt_arm is None or pred_arm is None:
            continue
        gt_curve = _coords_to_step_dict(gt_arm.get("coordinates", []))
        pred_curve = _coords_to_step_dict(pred_arm.get("coordinates", []))
        if not gt_curve:
            continue

        rmse, mae, pt_ok, pt_total, curve_score = _compare_step_curves(gt_curve, pred_curve)
        rmse_list.append(rmse)
        mae_list.append(mae)
        points_correct += pt_ok
        points_total += pt_total
        scores.append(curve_score)
        curve_details.append(
            {
                "gt_label": gt_arm.get("treatment_label"),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "points_within_tolerance": pt_ok,
                "points_total": pt_total,
                "curve_score": round(curve_score, 4),
            }
        )

    numeric = sum(scores) / len(scores) if scores else 0.0
    return (
        numeric,
        {"curves": curve_details, "scalar_scores": scores},
        _nanmean(rmse_list),
        _nanmean(mae_list),
        axis_rel_err,
        scalars_correct + points_correct,
        scalars_total + points_total,
    )


def _km_censoring_score(
    gt: dict,
    pred: dict,
    arm_pairs: List[Tuple[Optional[dict], Optional[dict]]],
) -> Tuple[float, dict]:
    """Set overlap of censoring tick times (tolerant)."""
    f1_scores = []
    details = []

    for gt_arm, pred_arm in arm_pairs:
        if gt_arm is None:
            continue
        gt_ticks = [float(t) for t in gt_arm.get("censoring_ticks", [])]
        pred_ticks = [float(t) for t in (pred_arm or {}).get("censoring_ticks", [])]
        p, r, f1 = _set_prf1(gt_ticks, pred_ticks, tol=_CENSOR_TIME_TOL)
        f1_scores.append(f1)
        details.append(
            {
                "gt_label": gt_arm.get("treatment_label"),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "gt_count": len(gt_ticks),
                "pred_count": len(pred_ticks),
            }
        )

    return (_nanmean(f1_scores) if f1_scores else 0.0), details


# ---------------------------------------------------------------------------
# Generic (non-KM) fallback
# ---------------------------------------------------------------------------


def _score_generic(
    gt: dict,
    pred: dict,
    type_match: bool,
    parse_err: Optional[str],
) -> ChartScore:
    """Recursive field comparison for forest / waterfall / anchor charts."""
    fields_ok, fields_total, mismatches = _compare_values(gt, pred, path="")
    field_ratio = fields_ok / fields_total if fields_total else 0.0
    type_score = 1.0 if type_match else 0.0
    overall = 0.1 * 1.0 + 0.1 * type_score + 0.8 * field_ratio

    return ChartScore(
        overall_score=round(overall, 4),
        json_valid=True,
        chart_type_match=type_match,
        parse_error=parse_err,
        text_score=field_ratio,
        structure_score=field_ratio,
        numeric_score=field_ratio,
        fields_correct=fields_ok,
        fields_total=fields_total,
        details={"mismatches_sample": mismatches[:20]},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dict(obj: JsonLike) -> Optional[dict]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        parsed, _ = extract_json_from_text(obj)
        return parsed
    return None


def _normalize_chart_type(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _text_similarity(a: str, b: str) -> float:
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s%()./-]", "", s)
    return s


def _nested_label(axes: dict, key: str) -> Optional[str]:
    node = axes.get(key)
    if isinstance(node, dict):
        return node.get("label")
    return None


def _nested_max(axes: dict, key: str) -> Optional[float]:
    node = axes.get(key)
    if isinstance(node, dict) and "max_value" in node:
        try:
            return float(node["max_value"])
        except (TypeError, ValueError):
            return None
    return None


def _match_arms_by_label(
    gt_arms: List[dict],
    pred_arms: List[dict],
) -> List[Tuple[Optional[dict], Optional[dict]]]:
    """Greedy one-to-one matching by treatment label similarity."""
    if not gt_arms:
        return []

    pred_remaining = list(pred_arms)
    pairs: List[Tuple[Optional[dict], Optional[dict]]] = []

    for gt_arm in gt_arms:
        gt_label = gt_arm.get("treatment_label", "")
        best_idx = -1
        best_sim = -1.0
        for i, pred_arm in enumerate(pred_remaining):
            sim = _text_similarity(gt_label, pred_arm.get("treatment_label", ""))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0 and best_sim >= 0.4:
            pairs.append((gt_arm, pred_remaining.pop(best_idx)))
        else:
            pairs.append((gt_arm, None))

    for pred_arm in pred_remaining:
        pairs.append((None, pred_arm))

    return pairs


def _coords_to_step_dict(coords: Sequence) -> Dict[float, float]:
    """KM step function: time -> survival probability (last value wins per time)."""
    curve: Dict[float, float] = {}
    for pt in coords:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            t, s = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        curve[t] = s
    return dict(sorted(curve.items()))


def _survival_at_time(curve: Dict[float, float], t: float) -> float:
    """Right-continuous step: value at largest time <= t."""
    if not curve:
        return 1.0
    val = 1.0
    for ts in sorted(curve.keys()):
        if ts <= t:
            val = curve[ts]
        else:
            break
    return val


def _compare_step_curves(
    gt_curve: Dict[float, float],
    pred_curve: Dict[float, float],
) -> Tuple[float, float, int, int, float]:
    """
    Compare GT step curve to prediction at all GT event times.

    Returns: rmse, mae, points_correct, points_total, curve_score
    """
    if not gt_curve:
        return 0.0, 0.0, 0, 0, 0.0

    sq_errs = []
    abs_errs = []
    correct = 0
    for t, gt_s in gt_curve.items():
        pred_s = _survival_at_time(pred_curve, t)
        err = abs(pred_s - gt_s)
        sq_errs.append(err * err)
        abs_errs.append(err)
        if err <= _SURV_ABS_TOL:
            correct += 1

    rmse = math.sqrt(sum(sq_errs) / len(sq_errs))
    mae = sum(abs_errs) / len(abs_errs)
    # Map RMSE to [0,1]: 0 error -> 1, RMSE >= 0.25 -> 0
    curve_score = max(0.0, 1.0 - rmse / 0.25)
    return rmse, mae, correct, len(gt_curve), curve_score


def _scalar_closeness(
    pred: float,
    gt: float,
    *,
    rel_tol: float,
    abs_tol: float,
) -> float:
    err = abs(pred - gt)
    if gt != 0:
        rel_err = err / abs(gt)
        if rel_err <= rel_tol:
            return 1.0
    if err <= abs_tol:
        return 1.0
    denom = max(abs_tol, abs(gt) * rel_tol, 1e-9)
    return max(0.0, 1.0 - err / denom)


def _set_prf1(
    gt_values: List[float],
    pred_values: List[float],
    *,
    tol: float,
) -> Tuple[float, float, float]:
    """Match predicted scalars to GT within tolerance (greedy)."""
    if not gt_values and not pred_values:
        return 1.0, 1.0, 1.0
    if not gt_values:
        return 0.0, 1.0, 0.0
    if not pred_values:
        return 0.0, 0.0, 0.0

    pred_used = [False] * len(pred_values)
    tp = 0
    for g in gt_values:
        best_j = -1
        best_dist = tol + 1
        for j, p in enumerate(pred_values):
            if pred_used[j]:
                continue
            d = abs(g - p)
            if d <= tol and d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0:
            pred_used[best_j] = True
            tp += 1

    precision = tp / len(pred_values) if pred_values else 0.0
    recall = tp / len(gt_values)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _values_close(a: Any, b: Any, *, rel_tol: float = 0.05, abs_tol: float = 1e-3) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    return abs(fa - fb) <= abs_tol


def _compare_values(
    gt: Any,
    pred: Any,
    *,
    path: str,
) -> Tuple[int, int, List[str]]:
    """Count leaf fields that match (with float tolerance)."""
    if isinstance(gt, dict):
        if not isinstance(pred, dict):
            return 0, max(1, len(gt)), [f"{path}: type mismatch"]
        ok, total, mismatches = 0, 0, []
        for k in gt:
            sub_path = f"{path}.{k}" if path else k
            o, t, m = _compare_values(gt[k], pred.get(k), path=sub_path)
            ok += o
            total += t
            mismatches.extend(m)
        return ok, total, mismatches

    if isinstance(gt, list):
        if not isinstance(pred, list):
            return 0, max(1, len(gt)), [f"{path}: type mismatch"]
        if len(gt) != len(pred):
            # Still compare min length element-wise
            pass
        ok, total, mismatches = 0, 0, []
        for i, gt_item in enumerate(gt):
            pred_item = pred[i] if i < len(pred) else None
            o, t, m = _compare_values(gt_item, pred_item, path=f"{path}[{i}]")
            ok += o
            total += t
            mismatches.extend(m)
        return ok, total, mismatches

    total = 1
    if isinstance(gt, (int, float)) and isinstance(pred, (int, float)):
        ok = 1 if _values_close(gt, pred) else 0
    elif isinstance(gt, str) and isinstance(pred, str):
        ok = 1 if _text_similarity(gt, pred) >= 0.85 else 0
    else:
        ok = 1 if gt == pred else 0
    mismatches = [] if ok else [f"{path}: {gt!r} vs {pred!r}"]
    return ok, total, mismatches


def _nanmean(values: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None and not math.isnan(v)]
    if not nums:
        return None
    return sum(nums) / len(nums)
