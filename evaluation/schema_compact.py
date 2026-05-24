"""
Compact chart JSON for Phase C training (fits 768-token budget).

Training uses minified labels; eval decompresses back to verbose schema for metrics.
"""

from __future__ import annotations

import json
from typing import List, Optional, Union

CHART_TYPE_COMPACT = {
    "kaplan_meier": "km",
    "forest_plot": "fp",
    "waterfall_plot": "wf",
    "simple_bar": "sb",
    "stacked_bar": "stb",
    "multi_line": "ml",
    "dual_axis_combo": "dac",
    "scatter": "sc",
}

CHART_TYPE_EXPAND = {v: k for k, v in CHART_TYPE_COMPACT.items()}

MAX_COORDS_PER_ARM = 10
MAX_CENSORS_PER_ARM = 6
MAX_SERIES_POINTS = 12
MAX_WATERFALL_BARS = 30


def _cap_evenly(items: list, max_n: int) -> list:
    if len(items) <= max_n:
        return items
    if max_n <= 1:
        return [items[0]]
    if max_n == 2:
        return [items[0], items[-1]]
    idxs = [int(i * (len(items) - 1) / (max_n - 1)) for i in range(max_n)]
    out, seen = [], set()
    for i in idxs:
        if i not in seen:
            out.append(items[i])
            seen.add(i)
    return out


def _round_coord(pt: list) -> list:
    return [round(float(pt[0]), 2), round(float(pt[1]), 4)]


def _round_float(v: float, digits: int = 2) -> float:
    return round(float(v), digits)


def _dedupe_sorted_floats(values: List[float]) -> List[float]:
    if not values:
        return []
    sorted_vals = sorted(float(v) for v in values)
    out = [sorted_vals[0]]
    for v in sorted_vals[1:]:
        if v != out[-1]:
            out.append(v)
    return out


def subsample_km_coordinates(coordinates: List[List[float]]) -> List[List[float]]:
    """Keep step-function corners: (0,1), survival drops, and final point."""
    if not coordinates:
        return []

    parsed: List[List[float]] = []
    for pt in coordinates:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        t, s = float(pt[0]), float(pt[1])
        parsed.append([t, max(0.0, min(1.0, s))])

    if not parsed:
        return []

    parsed.sort(key=lambda p: p[0])
    kept: List[List[float]] = []
    prev_s: Optional[float] = None

    for t, s in parsed:
        if not kept:
            kept.append([t, s])
            prev_s = s
            continue
        if s != prev_s:
            kept.append([t, s])
            prev_s = s

    last = parsed[-1]
    if kept[-1][0] != last[0] or kept[-1][1] != last[1]:
        kept.append([last[0], last[1]])

    deduped = [kept[0]]
    for p in kept[1:]:
        if p != deduped[-1]:
            deduped.append(p)
    return deduped


def minify_km(obj: dict) -> dict:
    axes = obj.get("axes", {})
    x_ax, y_ax = axes.get("x", {}), axes.get("y", {})
    arms_out = []
    for arm in obj.get("arms", []):
        coords = subsample_km_coordinates(arm.get("coordinates", []))
        coords = [_round_coord(p) for p in _cap_evenly(coords, MAX_COORDS_PER_ARM)]
        censors = _dedupe_sorted_floats(arm.get("censoring_ticks", []))
        censors = [_round_float(v) for v in _cap_evenly(censors, MAX_CENSORS_PER_ARM)]
        arms_out.append({"id": arm.get("treatment_label", ""), "p": coords, "c": censors})
    return {
        "ct": "km",
        "ax": {
            "x": {"l": x_ax.get("label", ""), "m": _round_float(x_ax.get("max_value", 0))},
            "y": {"l": y_ax.get("label", ""), "m": _round_float(y_ax.get("max_value", 1.0), 4)},
        },
        "a": arms_out,
    }


def decompress_km(obj: dict) -> dict:
    ax = obj.get("ax", {})
    x_ax, y_ax = ax.get("x", {}), ax.get("y", {})
    arms = []
    for arm in obj.get("a", []):
        if "t" in arm and "s" in arm:
            coords = [
                [float(t), float(s)]
                for t, s in zip(arm.get("t", []), arm.get("s", []))
            ]
        else:
            coords = [_round_coord(p) for p in arm.get("p", [])]
        arms.append(
            {
                "treatment_label": arm.get("id", ""),
                "coordinates": coords,
                "censoring_ticks": [float(v) for v in arm.get("c", [])],
            }
        )
    return {
        "chart_type": "kaplan_meier",
        "axes": {
            "x": {"label": x_ax.get("l", ""), "max_value": float(x_ax.get("m", 0))},
            "y": {"label": y_ax.get("l", ""), "max_value": float(y_ax.get("m", 1.0))},
        },
        "arms": arms,
    }


def minify_forest(obj: dict) -> dict:
    def study(s: dict) -> dict:
        return {
            "id": s.get("study_label", ""),
            "r": _round_float(s.get("ratio_value", 0), 3),
            "lo": _round_float(s.get("ci_lower", 0), 3),
            "hi": _round_float(s.get("ci_upper", 0), 3),
        }

    return {"ct": "fp", "ax": obj.get("axes", {}), "st": [study(s) for s in obj.get("studies", [])], "ov": study(obj.get("overall_effect", {}))}


def decompress_forest(obj: dict) -> dict:
    def study(s: dict) -> dict:
        return {
            "study_label": s.get("id", ""),
            "ratio_value": float(s.get("r", 0)),
            "ci_lower": float(s.get("lo", 0)),
            "ci_upper": float(s.get("hi", 0)),
        }

    return {
        "chart_type": "forest_plot",
        "axes": obj.get("ax", {}),
        "studies": [study(s) for s in obj.get("st", [])],
        "overall_effect": study(obj.get("ov", {})),
    }


def minify_waterfall(obj: dict) -> dict:
    bars = _cap_evenly(obj.get("bars", []), MAX_WATERFALL_BARS)
    return {
        "ct": "wf",
        "ax": obj.get("axes", {}),
        "b": [{"id": b.get("label", ""), "v": _round_float(b.get("value", 0), 3)} for b in bars],
    }


def decompress_waterfall(obj: dict) -> dict:
    return {
        "chart_type": "waterfall_plot",
        "axes": obj.get("ax", {}),
        "bars": [{"label": b.get("id", ""), "value": float(b.get("v", 0))} for b in obj.get("b", [])],
    }


def minify_anchor(obj: dict) -> dict:
    ct = obj.get("chart_type", "simple_bar")
    series = []
    for s in obj.get("series", []):
        data = _cap_evenly(s.get("data", []), MAX_SERIES_POINTS)
        series.append(
            {
                "id": s.get("series_name", ""),
                "k": s.get("series_type", "line"),
                "d": [{"x": p.get("x"), "y": _round_float(p.get("y", 0), 4)} for p in data],
            }
        )
    return {"ct": CHART_TYPE_COMPACT.get(ct, ct), "ax": obj.get("axes", {}), "s": series}


def decompress_anchor(obj: dict) -> dict:
    ct_short = obj.get("ct", "sb")
    series = []
    for s in obj.get("s", []):
        series.append(
            {
                "series_name": s.get("id", ""),
                "series_type": s.get("k", "line"),
                "data": [{"x": p.get("x"), "y": float(p.get("y", 0))} for p in s.get("d", [])],
            }
        )
    return {"chart_type": CHART_TYPE_EXPAND.get(ct_short, ct_short), "axes": obj.get("ax", {}), "series": series}


def minify_chart(obj: dict) -> dict:
    ct = obj.get("chart_type", "")
    if ct == "kaplan_meier":
        return minify_km(obj)
    if ct == "forest_plot":
        return minify_forest(obj)
    if ct == "waterfall_plot":
        return minify_waterfall(obj)
    if ct in CHART_TYPE_COMPACT:
        return minify_anchor(obj)
    if "ct" in obj and "chart_type" not in obj:
        return obj
    return obj


def decompress_chart(obj: dict) -> dict:
    if "chart_type" in obj:
        return obj
    ct = obj.get("ct", "")
    if ct == "km":
        return decompress_km(obj)
    if ct == "fp":
        return decompress_forest(obj)
    if ct == "wf":
        return decompress_waterfall(obj)
    if ct in CHART_TYPE_EXPAND:
        return decompress_anchor(obj)
    return obj


def compact_fits_token_budget(
    verbose_obj: dict,
    tokenizer,
    *,
    prompt: str = "\nExtract the underlying data from this clinical chart in strict JSON format.\n",
    max_length: int = 768,
) -> bool:
    text = prompt + compact_json_string(verbose_obj) + tokenizer.eos_token
    return len(tokenizer(text, add_special_tokens=True).input_ids) <= max_length


def precompressed_fits_token_budget(
    compressed_obj: dict,
    tokenizer,
    *,
    prompt: str = "\nExtract the underlying data from this clinical chart in strict JSON format.\n",
    max_length: int = 768,
    use_chatml: bool = False,
) -> bool:
    target = json.dumps(compressed_obj, separators=(",", ":"))
    text = build_training_text(prompt, target, tokenizer, use_chatml=use_chatml)
    return len(tokenizer(text, add_special_tokens=True).input_ids) <= max_length


def build_training_text(
    user_prompt: str,
    target_json: str,
    tokenizer,
    *,
    use_chatml: bool = False,
) -> str:
    """Plain prompt+JSON or Qwen ChatML (user/assistant) training string."""
    if not use_chatml:
        return user_prompt + target_json + tokenizer.eos_token
    user_block = user_prompt.strip()
    messages = [
        {"role": "user", "content": user_block},
        {"role": "assistant", "content": target_json},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ) + tokenizer.eos_token
    except Exception:
        return (
            f"<|im_start|>user\n{user_block}\n\n"
            f"<|im_start|>assistant\n{target_json}\n"
        )


def prompt_mask_length(
    user_prompt: str,
    tokenizer,
    *,
    use_chatml: bool = False,
) -> int:
    """Token length to mask (prompt only, excluding target JSON)."""
    if not use_chatml:
        return tokenizer(user_prompt, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
    user_block = user_prompt.strip()
    try:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_block}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prefix = f"<|im_start|>user\n{user_block}\n\n<|im_start|>assistant\n"
    return len(tokenizer(prefix, add_special_tokens=True).input_ids)


def compact_json_string(verbose_obj: dict) -> str:
    return json.dumps(minify_chart(verbose_obj), separators=(",", ":"))


def try_decompress_prediction(pred: Union[dict, str]) -> dict:
    if isinstance(pred, str):
        from evaluation.parse_output import extract_json_from_text

        parsed, _ = extract_json_from_text(pred)
        if parsed is None:
            raise ValueError("could not parse prediction JSON")
        pred = parsed
    return decompress_chart(pred)
