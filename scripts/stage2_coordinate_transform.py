"""

Stage 2 coordinate transforms: clinical <-> normalized local tile space.



Normalized labels use [x_norm, y_norm] in [0, 1] relative to the 384x384 tile

(top-left = 0,0; bottom-right = 1,1). Middleware uses _meta to map back to clinical.

"""



from __future__ import annotations



import sys

from pathlib import Path

from typing import List, Sequence, Tuple



ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))



from stage2_common import coords_to_pairs





def normalized_local_to_clinical(

    x_norm: float,

    y_norm: float,

    *,

    tile_origin: Sequence[int],

    plot_bbox: Sequence[int],

    axis_max: dict,

    tile_size: int = 384,

) -> Tuple[float, float]:

    """

    Inverse of generate_stage2_tiles pixel pipeline.



    plot_bbox: [x0, y0, x1, y1] on 768x768 canvas

    tile_origin: [tx0, ty0] top-left of crop

    axis_max: {"x": x_max, "y": y_max}

    """

    tx0, ty0 = int(tile_origin[0]), int(tile_origin[1])

    px0, py0, px1, py1 = (int(v) for v in plot_bbox)

    x_max = float(axis_max.get("x", 1.0) or 1.0)

    y_max = float(axis_max.get("y", 1.0) or 1.0)

    plot_w = max(1, px1 - px0)

    plot_h = max(1, py1 - py0)



    x_local = float(x_norm) * tile_size

    y_local = float(y_norm) * tile_size

    px = tx0 + x_local

    py = ty0 + y_local



    t = (px - px0) / plot_w * x_max

    s = (py1 - py) / plot_h * y_max

    return round(t, 4), round(max(0.0, min(1.0, s)), 6)





def normalized_points_to_clinical(label_obj: dict) -> List[List[float]]:

    """Convert a normalized_local tile label to clinical [t,s] pairs."""

    meta = label_obj.get("_meta", {})

    if meta.get("coordinate_space") != "normalized_local":

        return coords_to_pairs(label_obj.get("points", []))

    out = []

    for pair in coords_to_pairs(label_obj.get("points", [])):

        t, s = normalized_local_to_clinical(

            pair[0],

            pair[1],

            tile_origin=meta["tile_origin"],

            plot_bbox=meta["plot_bbox"],

            axis_max=meta["axis_max"],

        )

        out.append([t, s])

    return out


