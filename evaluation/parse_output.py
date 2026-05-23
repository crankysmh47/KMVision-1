"""Robust JSON extraction from model-generated text."""

import json
import re
from typing import Any, Optional, Tuple

try:
    import json_repair
except ImportError:
    json_repair = None  # type: ignore


def repair_truncated_chart_json(text: str) -> str:
    """
    Fix outputs that begin mid-schema (common when decoding only new tokens).
    Example: axes": { ...  ->  {"chart_type": "kaplan_meier", "axes": { ...
    """
    t = text.strip()
    if not t:
        return t
    if t.startswith("{"):
        return t
    if t.startswith('axes"') or t.startswith("axes"):
        return '{"chart_type": "kaplan_meier", "' + t
    return t


def extract_json_from_text(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Parse a JSON object from model output that may include markdown fences or prose.

    Returns:
        (parsed_dict, error_message) — dict is None on failure.
    """
    if not text or not text.strip():
        return None, "empty output"

    cleaned = repair_truncated_chart_json(text.strip())

    # Strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, None
        return None, "parsed value is not a JSON object"
    except json.JSONDecodeError:
        pass

    # Find first balanced {...} object
    start = cleaned.find("{")
    if start == -1:
        return None, "no JSON object found"

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj, None
                except json.JSONDecodeError as e:
                    return None, f"invalid JSON object: {e}"

    if json_repair is not None:
        try:
            repaired = repair_truncated_chart_json(cleaned)
            obj = json.loads(json_repair.repair_json(repaired))
            if isinstance(obj, dict):
                return obj, None
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return None, "unbalanced or truncated JSON object"
