"""Shared Stage 2 prompt, JSON prefix forcing, and label-mask helpers."""



from __future__ import annotations



import json

from typing import Any, List, Sequence, Tuple



STAGE2_PROMPT_TEMPLATE = (

    "\nExtract dense coordinates and censoring ticks for the '{arm_id}' "

    "line in this 384x384 tile. Output JSON with flat lists: "

    '"points": [x1, y1, x2, y2, ...] and "censors": [x1, y1, ...] '

    "using normalized local values in [0.000, 1.000] where (0,0) is the "

    "top-left of the tile and (1,1) is the bottom-right. "

    "Do not nest [x,y] pairs inside the arrays.\n"

)



COORDINATE_SPACE_NORMALIZED = "normalized_local"

POINT_FORMAT_FLAT = "flat_xy"

COORD_DECIMALS = 3





def stage2_user_prompt(arm_id: str) -> str:

    return STAGE2_PROMPT_TEMPLATE.format(arm_id=arm_id)





def force_json_assistant_prefix(arm_id: str) -> str:

    """Pre-fill assistant JSON through arm_id and opening of points array."""

    arm_lit = json.dumps(str(arm_id), ensure_ascii=False)

    return f'{{"arm_id": {arm_lit}, "points": ['





def pairs_to_flat_xy(pairs: Sequence[Any]) -> List[float]:

    """[[x,y],...] -> [x,y,x,y,...] rounded to 3 decimals."""

    out: List[float] = []

    for p in pairs:

        if isinstance(p, (list, tuple)) and len(p) >= 2:

            out.append(round(float(p[0]), COORD_DECIMALS))

            out.append(round(float(p[1]), COORD_DECIMALS))

    return out





def flat_xy_to_pairs(flat: Sequence[Any]) -> List[List[float]]:

    """[x,y,x,y,...] -> [[x,y],...]; drops trailing orphan."""

    if not flat:

        return []

    vals = [float(v) for v in flat]

    if len(vals) % 2:

        vals = vals[:-1]

    return [[vals[i], vals[i + 1]] for i in range(0, len(vals), 2)]





def coords_to_pairs(raw: Any) -> List[List[float]]:

    """Accept nested pairs or flat interleaved lists."""

    if not isinstance(raw, list) or not raw:

        return []

    if isinstance(raw[0], (list, tuple)):

        return flat_xy_to_pairs(pairs_to_flat_xy(raw))

    if isinstance(raw[0], (int, float)):

        return flat_xy_to_pairs(raw)

    return []





def stage2_target_payload(label_obj: dict) -> dict:

    """Training/eval target: flat points/censors (no _meta)."""

    return {

        "arm_id": label_obj["arm_id"],

        "points": pairs_to_flat_xy(label_obj.get("points", [])),

        "censors": pairs_to_flat_xy(label_obj.get("censors", [])),

    }





def stage2_target_json(label_obj: dict) -> str:

    return json.dumps(stage2_target_payload(label_obj), separators=(",", ":"))





def chat_prefix_for_user(user_prompt: str, tokenizer) -> str:

    try:

        return tokenizer.apply_chat_template(

            [{"role": "user", "content": user_prompt.strip()}],

            tokenize=False,

            add_generation_prompt=True,

        )

    except Exception:

        return f"<|im_start|>user\n{user_prompt.strip()}\n\n<|im_start|>assistant\n"





def _encode_token_len(tokenizer, text: str) -> int:

    """Token count for a single string (handles flat or nested input_ids)."""

    ids = tokenizer(text, add_special_tokens=True)["input_ids"]

    if ids and isinstance(ids[0], list):

        return len(ids[0])

    return len(ids)





def mask_len_through_json_prefix(user_prompt: str, arm_id: str, tokenizer) -> int:

    """

    Token length to mask (no loss): user ChatML block + forced JSON prefix

    through '{"arm_id": "...", "points": ['.

    """

    partial = chat_prefix_for_user(user_prompt, tokenizer) + force_json_assistant_prefix(arm_id)

    return _encode_token_len(tokenizer, partial)





def as_point_tuples(raw: Any) -> List[Tuple[float, float]]:

    """Parse points field whether nested or flat_xy."""

    return [(p[0], p[1]) for p in coords_to_pairs(raw)]


