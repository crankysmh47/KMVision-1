"""Robust JSON extraction from model-generated text."""

import json
import re
from typing import Any, Optional, Tuple

try:
    import json_repair
except ImportError:
    json_repair = None  # type: ignore

# Stage 2 flat coordinate arrays: "points": [x1, y1, ...], "censors": [x1, y1, ...]
_ARRAY_FIELD_RE = re.compile(r'"(points|censors)"\s*:\s*\[')
_LEADING_COMMA_RE = re.compile(r'(\[\s*),(\s*[-\d.])')
_TRAILING_COMMA_RE = re.compile(r',(\s*])')
_ORPHAN_INT_RE = re.compile(
    r'("(?:points|censors)"\s*:\s*\[\s*)(-?\d+)(\s*,\s*(?:0\.\d+|1(?:\.0+)?)\s*,)',
)


def repair_stage2_json(text: str) -> str:
    """
    Fix common Stage 2 flat_xy malformations before strict JSON parsing.

    Handles:
    - Leading commas: "points": [, 0.299, ...]
    - Trailing commas before ]
    - Orphan integer at array start: "points": [9, 0.144, ...] (truncation artifact)
    - Missing "censors" key when points array is present
    - Truncated tail: close open arrays/object when braces are unbalanced
    """
    t = text.strip()
    if not t:
        return t

    # Ensure object wrapper when prefix-forced generation starts mid-stream.
    if not t.startswith("{"):
        if '"points"' in t or t.startswith('"arm_id"'):
            t = "{" + t
        elif t.startswith("arm_id"):
            t = '{"' + t

    # Normalize array comma artifacts inside points/censors lists.
    for _ in range(8):
        prev = t
        t = _LEADING_COMMA_RE.sub(r"\1\2", t)
        t = _TRAILING_COMMA_RE.sub(r"\1", t)
        if t == prev:
            break

    # Drop orphan leading integer when followed by normalized float pairs.
    def _drop_orphan(match: re.Match[str]) -> str:
        prefix, orphan, rest = match.group(1), match.group(2), match.group(3)
        try:
            val = int(orphan)
        except ValueError:
            return match.group(0)
        # Keep 0/1 when they are valid normalized coordinates.
        if val in (0, 1):
            return match.group(0)
        if abs(val) >= 2:
            return prefix + rest.lstrip(", ")
        return match.group(0)

    t = _ORPHAN_INT_RE.sub(_drop_orphan, t)

    # Close open arrays before adding missing keys or object braces.
    open_brackets = t.count("[") - t.count("]")
    if open_brackets > 0:
        t += "]" * open_brackets

    # Default missing censors when points are present.
    if '"points"' in t and '"censors"' not in t:
        if t.rstrip().endswith("}"):
            t = t.rstrip()[:-1].rstrip().rstrip(",") + ', "censors": []}'
        else:
            t = t.rstrip().rstrip(",") + ', "censors": []}'

    if t.count("{") > t.count("}"):
        t += "}" * (t.count("{") - t.count("}"))

    return t


def normalize_stage2_dict(obj: dict) -> dict:
    """Ensure arm_id/points/censors keys with list values after parse."""
    out = dict(obj)
    if "points" not in out:
        out["points"] = []
    if "censors" not in out:
        out["censors"] = []
    if not isinstance(out.get("points"), list):
        out["points"] = []
    if not isinstance(out.get("censors"), list):
        out["censors"] = []
    return out


def extract_stage2_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Parse Stage 2 tile output {"arm_id", "points", "censors"} with repair heuristics.
    """
    if not text or not text.strip():
        return None, "empty output"

    cleaned = repair_stage2_json(text.strip())
    parsed, err = extract_json_from_text(cleaned)
    if isinstance(parsed, dict) and ("points" in parsed or "censors" in parsed or "arm_id" in parsed):
        return normalize_stage2_dict(parsed), None
    return parsed, err


def repair_truncated_chart_json(text: str) -> str:
    """
    Fix outputs that begin mid-schema (common when decoding only new tokens).

    Verbose example: axes": { ...  ->  {"chart_type": "kaplan_meier", "axes": { ...
    Minified KM example: Perm...","m":1.0},"y":...,"a":[...  -> prepend ct/ax header.
    """
    t = text.strip()
    if not t:
        return t
    if t.startswith("{"):
        return t
    if t.startswith('axes"') or t.startswith("axes"):
        return '{"chart_type": "kaplan_meier", "' + t
    # Phase C minified KM: header {"ct":"km","ax":{"x":{"l":" was sliced off.
    if re.search(r'"a"\s*:\s*\[', t) and re.search(r'"(p|id)"\s*:', t):
        return '{"ct":"km","ax":{"x":{"l":"' + t
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
    if '"points"' in cleaned or '"censors"' in cleaned:
        cleaned = repair_stage2_json(cleaned)

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
