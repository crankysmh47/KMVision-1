"""Robust JSON extraction from model-generated text."""

import json
import re
from typing import Any, List, Optional, Tuple

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
# Bare decimals the tokenizer emits without a leading zero: [.623, 0.015] or [0.1, .2]
_BARE_DECIMAL_RE = re.compile(r'([\[,]\s*)\.(\d)')
# Leading-zero integer literals (invalid JSON): [01, 0.218] or [007, ...]
_LEADING_ZERO_INT_RE = re.compile(r'([\[,]\s*)0+(\d)(?![.\d])')
# Stray empty string between comma and next key: ]," "censors":  ->  ],"censors":
_STRAY_EMPTY_STRING_RE = re.compile(r',\s*"\s*"\s*"')
# Restart pathology: model begins a second concatenated object {"arm_id": ...
_ARM_ID_RE = re.compile(r'"arm_id"')


def repair_stage2_json(text: str) -> str:
    """
    Fix common Stage 2 flat_xy malformations before strict JSON parsing.

    Handles:
    - Leading commas: "points": [, 0.299, ...]
    - Trailing commas before ]
    - Orphan integer at array start: "points": [9, 0.144, ...] (truncation artifact)
    - Bare decimals: [.623, ...] -> [0.623, ...]
    - Leading-zero integers: [01, ...] -> [1, ...]
    - Stray string token between comma and key: ]," "censors": -> ],"censors":
    - Restart pathology: model emits a second concatenated {"arm_id": ... object;
      keep only the first object
    - Missing "censors" key when points array is present
    - Truncated/mismatched tails: repair bracket structure in LIFO order
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

    # Restart pathology: keep only the first object when the model starts a
    # second concatenated {"arm_id": ...} after finishing the first.
    arm_id_positions = [m.start() for m in _ARM_ID_RE.finditer(t)]
    if len(arm_id_positions) >= 2:
        brace = t.rfind("{", 0, arm_id_positions[1])
        if brace > 0:
            t = t[:brace].rstrip().rstrip(",").rstrip()

    # Stray blank string between a comma and the next (unquoted) key:
    # ]," "censors": -> ],"censors":
    # Normal ],"censors": is untouched (second quote position holds a letter).
    t = re.sub(r',\s*"\s*"\s*(?=\w)', ',"', t)

    # Normalize invalid numeric literals before comma/orphan repairs:
    # bare decimals (.623 -> 0.623) and leading-zero integers (01 -> 1).
    t = _BARE_DECIMAL_RE.sub(r"\g<1>0.\g<2>", t)
    t = _LEADING_ZERO_INT_RE.sub(r"\g<1>\g<2>", t)

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

    # Close any container still open at the correct nesting depth, then
    # default missing keys. (Order matters: the old count-based repair
    # appended "]" after a trailing "}", producing invalid JSON.)
    t = _close_open_containers(t)

    # Default missing censors when points are present.
    if '"points"' in t and '"censors"' not in t:
        if t.rstrip().endswith("}"):
            t = t.rstrip()[:-1].rstrip().rstrip(",") + ', "censors": []}'
        else:
            t = t.rstrip().rstrip(",") + ', "censors": []}'

    t = _close_open_containers(t)
    return t


def _close_open_containers(text: str) -> str:
    """Repair bracket/brace structure: fix mismatched closers, drop stray
    closers, and append missing closers at the correct nesting depth.

    String-aware single scan. Examples:
      {"points": [0.1, 0.2}      -> {"points": [0.1, 0.2]}
      {"points": [0.1, 0.2       -> {"points": [0.1, 0.2]}
      {"censors": [0.9, 0.9]}    -> unchanged
      {"points": [0.1]]}         -> {"points": [0.1]}  (stray "]" dropped)

    A naive count-based repair appends "]" after a trailing "}", producing
    invalid JSON; this one fixes closers in place and closes in LIFO order.
    """
    out: List[str] = []
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            out.append(ch)
            continue
        if ch in "[{":
            stack.append(ch)
            out.append(ch)
        elif ch in "]}":
            if not stack:
                continue  # stray extra closer at depth 0: drop it
            expected = "]" if stack[-1] == "[" else "}"
            out.append(expected)  # fix mismatched closer in place
            stack.pop()
        else:
            out.append(ch)
    for opener in reversed(stack):
        out.append("]" if opener == "[" else "}")
    return "".join(out)


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
