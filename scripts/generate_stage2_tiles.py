"""
Generate Stage 2 training tiles from dense Phase B synthetic KM charts.

Reads global chart images + verbose JSON labels, slides 384x384 windows across the
estimated plot area, and emits per-arm tile/label pairs with dense local points.

See docs/STAGE2_DECISIONS.md for design rationale.

Usage (do not run in CI by default):
  python scripts/generate_stage2_tiles.py --max_charts 500
  python scripts/generate_stage2_tiles.py --dataset_root "C:\\sem4\\KMVision-1 Data\\dataset"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.schema_compact import _cap_evenly, subsample_km_coordinates
from stage2_common import COORDINATE_SPACE_NORMALIZED

DEFAULT_DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"
CATEGORY = "km"
TARGET_SIZE = 768
TILE_SIZE = 384
TILE_OVERLAP = 50
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP
MIN_POINTS_PER_TILE = 2
MAX_POINTS_PER_TILE = 40
MAX_CENSORS_PER_TILE = 10
COORD_DECIMALS = 3
WHITE_THRESHOLD = 245
MARGIN_SHRINK = 8
FALLBACK_FRAC = (0.14, 0.12, 0.96, 0.22)  # left, top, right, bottom (1 - bottom = bottom inset)


@dataclass(frozen=True)
class PlotBBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class TileWindow:
    x0: int
    y0: int

    @property
    def x1(self) -> int:
        return self.x0 + TILE_SIZE

    @property
    def y1(self) -> int:
        return self.y0 + TILE_SIZE


def _slug(text: str, max_len: int = 32) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return (s or "arm")[:max_len]


def load_km_label(path: Path) -> dict:
    with open(path, encoding="utf-8", errors="replace") as f:
        obj = json.load(f)
    if obj.get("chart_type") != "kaplan_meier":
        raise ValueError(f"Not a KM chart: {path}")
    return obj


def normalize_image_768(image: Image.Image) -> Image.Image:
    """Resize preserving aspect via letterbox pad to 768x768 (DECISIONS §1)."""
    image = image.convert("RGB")
    w, h = image.size
    if w == TARGET_SIZE and h == TARGET_SIZE:
        return image
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (255, 255, 255))
    ox, oy = (TARGET_SIZE - nw) // 2, (TARGET_SIZE - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def estimate_plot_bbox(image: Image.Image) -> PlotBBox:
    """
    Infer plot area from raster (synthetic JSON has no pixel axes metadata).
    See docs/STAGE2_DECISIONS.md §1.
    """
    arr = np.asarray(image.convert("L"))
    mask = arr < WHITE_THRESHOLD
    if not mask.any():
        return _fallback_bbox(arr.shape[1], arr.shape[0])

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    x0 = min(max(0, x0 + MARGIN_SHRINK), TARGET_SIZE - 1)
    y0 = min(max(0, y0 + MARGIN_SHRINK), TARGET_SIZE - 1)
    x1 = max(min(TARGET_SIZE, x1 - MARGIN_SHRINK), x0 + 1)
    y1 = max(min(TARGET_SIZE, y1 - MARGIN_SHRINK), y0 + 1)

    if (x1 - x0) < 128 or (y1 - y0) < 128:
        return _fallback_bbox(arr.shape[1], arr.shape[0])
    return PlotBBox(x0, y0, x1, y1)


def _fallback_bbox(width: int, height: int) -> PlotBBox:
    left, top, right, bottom = FALLBACK_FRAC
    return PlotBBox(
        int(width * left),
        int(height * top),
        int(width * right),
        int(height * (1.0 - bottom)),
    )


def axis_limits(chart: dict) -> Tuple[float, float]:
    axes = chart.get("axes", {})
    x_max = float(axes.get("x", {}).get("max_value", 1.0) or 1.0)
    y_max = float(axes.get("y", {}).get("max_value", 1.0) or 1.0)
    if x_max <= 0:
        x_max = 1.0
    if y_max <= 0:
        y_max = 1.0
    return x_max, y_max


def clinical_to_pixel(
    t: float,
    s: float,
    plot: PlotBBox,
    x_max: float,
    y_max: float,
) -> Tuple[float, float]:
    px = plot.x0 + (t / x_max) * plot.width
    py = plot.y1 - (s / y_max) * plot.height
    return px, py


def pixel_in_tile(px: float, py: float, tile: TileWindow) -> bool:
    return tile.x0 <= px < tile.x1 and tile.y0 <= py < tile.y1


def survival_at_time(coordinates: Sequence[Sequence[float]], t: float) -> float:
    """Right-continuous KM step value at time t."""
    pts = sorted((float(p[0]), float(p[1])) for p in coordinates if len(p) >= 2)
    if not pts:
        return 0.0
    if t <= pts[0][0]:
        return pts[0][1]
    surv = pts[0][1]
    for t_i, s_i in pts[1:]:
        if t < t_i:
            return surv
        surv = s_i
    return pts[-1][1]


def iter_tile_windows(plot: PlotBBox) -> Iterator[TileWindow]:
    """Horizontal sweep with overlap; vertical band centered in plot (DECISIONS §3)."""
    ty0 = plot.y0 + max(0, (plot.height - TILE_SIZE) // 2)
    ty1 = ty0 + TILE_SIZE
    if ty1 > plot.y1:
        ty0 = max(plot.y0, plot.y1 - TILE_SIZE)
        ty0 = max(0, ty0)

    x = plot.x0
    while x + TILE_SIZE <= plot.x1 + TILE_STRIDE:
        x_clamped = min(x, max(0, TARGET_SIZE - TILE_SIZE))
        yield TileWindow(x_clamped, ty0)
        if x_clamped + TILE_SIZE >= plot.x1:
            break
        x += TILE_STRIDE


def pixel_to_normalized_local(px: float, py: float, tile: TileWindow) -> List[float]:
    """Map global 768-space pixel to [0,1] tile coords; (0,0)=top-left, (1,1)=bottom-right."""
    x_local = px - tile.x0
    y_local = py - tile.y0
    x_norm = round(max(0.0, min(1.0, x_local / TILE_SIZE)), COORD_DECIMALS)
    y_norm = round(max(0.0, min(1.0, y_local / TILE_SIZE)), COORD_DECIMALS)
    return [x_norm, y_norm]


def clinical_pair_to_normalized_local(
    t: float,
    s: float,
    tile: TileWindow,
    plot: PlotBBox,
    x_max: float,
    y_max: float,
) -> List[float]:
    px, py = clinical_to_pixel(t, s, plot, x_max, y_max)
    return pixel_to_normalized_local(px, py, tile)


def filter_arm_points_clinical(
    arm: dict,
    tile: TileWindow,
    plot: PlotBBox,
    x_max: float,
    y_max: float,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Points/censors in clinical (time, survival) that fall inside the tile."""
    coords = arm.get("coordinates", [])
    points: List[List[float]] = []
    for pt in coords:
        if len(pt) < 2:
            continue
        t, s = float(pt[0]), float(pt[1])
        px, py = clinical_to_pixel(t, s, plot, x_max, y_max)
        if pixel_in_tile(px, py, tile):
            points.append([round(t, 4), round(max(0.0, min(1.0, s)), 6)])

    censors: List[List[float]] = []
    for tick in arm.get("censoring_ticks", []):
        t_c = float(tick)
        s_c = survival_at_time(coords, t_c)
        px, py = clinical_to_pixel(t_c, s_c, plot, x_max, y_max)
        if pixel_in_tile(px, py, tile):
            censors.append([round(t_c, 4), round(s_c, 6)])

    seen = set()
    uniq_points = []
    for p in points:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            uniq_points.append(p)
    return uniq_points, censors


def clinical_lists_to_tile_space(
    points_clinical: List[List[float]],
    censors_clinical: List[List[float]],
    tile: TileWindow,
    plot: PlotBBox,
    x_max: float,
    y_max: float,
    *,
    coordinate_space: str,
) -> Tuple[List[List[float]], List[List[float]]]:
    if coordinate_space == COORDINATE_SPACE_NORMALIZED:
        points = [
            clinical_pair_to_normalized_local(t, s, tile, plot, x_max, y_max)
            for t, s in points_clinical
        ]
        censors = [
            clinical_pair_to_normalized_local(t, s, tile, plot, x_max, y_max)
            for t, s in censors_clinical
        ]
        return points, censors
    return points_clinical, censors_clinical


def cap_tile_points(points: List[List[float]]) -> List[List[float]]:
    """Step-aware subsample (Phase C logic on clinical t,s), then evenly cap."""
    stepwise = subsample_km_coordinates(points)
    return _cap_evenly(stepwise, MAX_POINTS_PER_TILE)


def cap_tile_censors(censors: List[List[float]]) -> List[List[float]]:
    """Cap censor [t,s] pairs to MAX_CENSORS_PER_TILE, evenly spaced in time."""
    if len(censors) <= MAX_CENSORS_PER_TILE:
        return censors
    sorted_c = sorted(censors, key=lambda p: p[0])
    return _cap_evenly(sorted_c, MAX_CENSORS_PER_TILE)


def time_window_from_tile(
    tile: TileWindow, plot: PlotBBox, x_max: float
) -> Tuple[float, float]:
    t_lo = (tile.x0 - plot.x0) / plot.width * x_max
    t_hi = (tile.x1 - plot.x0) / plot.width * x_max
    return round(max(0.0, t_lo), 4), round(max(0.0, t_hi), 4)


def process_chart(
    img_path: Path,
    label_path: Path,
    out_images: Path,
    out_labels: Path,
    *,
    stem_prefix: str = "",
    coordinate_space: str = COORDINATE_SPACE_NORMALIZED,
) -> int:
    chart = load_km_label(label_path)
    image = Image.open(img_path)
    image = normalize_image_768(image)
    plot = estimate_plot_bbox(image)
    x_max, y_max = axis_limits(chart)

    saved = 0
    source_stem = stem_prefix or label_path.stem

    for tile_idx, tile in enumerate(iter_tile_windows(plot)):
        crop = image.crop((tile.x0, tile.y0, tile.x1, tile.y1))
        if crop.size != (TILE_SIZE, TILE_SIZE):
            crop = crop.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)

        t_lo, t_hi = time_window_from_tile(tile, plot, x_max)

        for arm_idx, arm in enumerate(chart.get("arms", [])):
            arm_id = str(arm.get("treatment_label", f"arm_{arm_idx}"))
            points_clin, censors_clin = filter_arm_points_clinical(
                arm, tile, plot, x_max, y_max
            )
            if len(points_clin) < MIN_POINTS_PER_TILE:
                continue

            n_raw_points = len(points_clin)
            n_raw_censors = len(censors_clin)
            points_clin = cap_tile_points(points_clin)
            censors_clin = cap_tile_censors(censors_clin)
            points, censors = clinical_lists_to_tile_space(
                points_clin,
                censors_clin,
                tile,
                plot,
                x_max,
                y_max,
                coordinate_space=coordinate_space,
            )

            base = f"{source_stem}_x{tile.x0:04d}_arm{arm_idx}_{_slug(arm_id)}"
            img_out = out_images / f"{base}.png"
            lbl_out = out_labels / f"{base}.json"

            label_obj = {
                "arm_id": arm_id,
                "points": points,
                "censors": censors,
                "_meta": {
                    "coordinate_space": coordinate_space,
                    "coord_decimals": COORD_DECIMALS,
                    "source_chart": source_stem,
                    "source_image": str(img_path),
                    "tile_index": tile_idx,
                    "tile_origin": [tile.x0, tile.y0],
                    "time_window": [t_lo, t_hi],
                    "plot_bbox": [plot.x0, plot.y0, plot.x1, plot.y1],
                    "axis_max": {"x": x_max, "y": y_max},
                    "points_before_cap": n_raw_points,
                    "censors_before_cap": n_raw_censors,
                    "max_points_per_tile": MAX_POINTS_PER_TILE,
                    "max_censors_per_tile": MAX_CENSORS_PER_TILE,
                },
            }
            crop.save(img_out)
            with open(lbl_out, "w", encoding="utf-8") as f:
                json.dump(label_obj, f, indent=2)
            saved += 1

    return saved


def collect_pairs(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    if not label_dir.is_dir():
        return pairs
    for label_path in sorted(label_dir.glob("*.json")):
        stem = label_path.stem
        for ext in (".png", ".jpg"):
            img_path = image_dir / f"{stem}{ext}"
            if img_path.is_file():
                pairs.append((img_path, label_path))
                break
    return pairs


def resolve_source_dirs(root: Path, source: str) -> Tuple[Path, Path]:
    """
    testing = holdout never used in Phase B/C (organize_train_test.py).
    train_1 = Phase B/C pool — avoid for Stage 2 unless explicitly requested.
    """
    if source == "testing":
        return root / "testing" / "images" / CATEGORY, root / "testing" / "labels" / CATEGORY
    if source == "train_1":
        return root / "train_1" / "images" / CATEGORY, root / "train_1" / "labels" / CATEGORY
    raise ValueError(f"Unknown --source {source!r}; use 'testing' or 'train_1'")


def load_phase_train_stems(root: Path, manifest_path: Optional[Path]) -> Set[str]:
    """
    Stems used in Phase B/C training — exclude from Stage 2 tile sources.
    Uses train_1/ on disk plus optional split_manifest.json train list.
    """
    excluded: Set[str] = set()
    train_lbl = root / "train_1" / "labels" / CATEGORY
    if train_lbl.is_dir():
        excluded.update(p.stem for p in train_lbl.glob("*.json"))

    if manifest_path and manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("train", []):
            label = entry.get("label", "")
            if label:
                excluded.add(Path(label).stem)

    return excluded


def filter_unused_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    excluded_stems: Set[str],
) -> List[Tuple[Path, Path]]:
    if not excluded_stems:
        return list(pairs)
    kept = [(i, l) for i, l in pairs if Path(l).stem not in excluded_stems]
    dropped = len(pairs) - len(kept)
    if dropped:
        print(f"Excluded {dropped} charts overlapping Phase B/C train_1 / manifest train.")
    return kept


def clear_tile_output(base: Path) -> None:
    for sub in ("images", "labels"):
        path = base / sub / CATEGORY
        if path.is_dir():
            shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Stage 2 tile dataset from dense KM synthetic data.")
    p.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--source",
        type=str,
        default="testing",
        choices=["testing", "train_1"],
        help="testing = holdout not used in Phase B/C (default). train_1 = Phase B/C pool.",
    )
    p.add_argument("--image_dir", type=str, default=None, help="Override image dir (ignores --source).")
    p.add_argument("--label_dir", type=str, default=None, help="Override label dir (ignores --source).")
    p.add_argument("--split_manifest", type=str, default=None, help="Default: {root}/split_manifest.json")
    p.add_argument("--output_dir", type=str, default=None, help="Default: {root}/stage2_v2_1")
    p.add_argument("--holdout_dir", type=str, default=None, help="Default: {root}/stage2_v2_1_holdout")
    p.add_argument(
        "--coordinate-space",
        type=str,
        default=COORDINATE_SPACE_NORMALIZED,
        choices=[COORDINATE_SPACE_NORMALIZED, "clinical"],
        help="Label coordinate system (default: normalized_local 0-1 tile space).",
    )
    p.add_argument("--holdout_fraction", type=float, default=0.05)
    p.add_argument("--max_charts", type=int, default=12000, help="Cap source charts (default 12000).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--clear_output",
        action="store_true",
        help="Remove existing stage2/ and stage2_holdout/ tile trees before writing.",
    )
    p.add_argument(
        "--chart_ids_file",
        type=str,
        default=None,
        help="Path to a text file of chart IDs; generates tiles ONLY for these charts "
             "into --output_dir (no holdout split). Intended for validation sets.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root)

    # --chart-ids-file mode: generate tiles ONLY for the listed chart IDs
    # (all into one output dir, no holdout split). Used for validation sets.
    if getattr(args, "chart_ids_file", None):
        ids = [
            line.strip()
            for line in Path(args.chart_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        id_set = set(ids)
        image_dir = Path(args.image_dir) if args.image_dir else root / "testing" / "images" / "km"
        label_dir = Path(args.label_dir) if args.label_dir else root / "testing" / "labels" / "km"
        out_root = Path(args.output_dir or root / "stage2_validation")
        img_out = out_root / "images" / CATEGORY
        lbl_out = out_root / "labels" / CATEGORY
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        total = 0
        skipped = 0
        done = 0
        for lf in sorted(label_dir.glob("*.json")):
            if lf.stem not in id_set:
                continue
            stem = lf.stem  # e.g. chart_0026be92_km -> image file same stem
            img_path = image_dir / f"{stem}.png"
            if not img_path.is_file():
                skipped += 1
                continue
            try:
                total += process_chart(img_path, lf, img_out, lbl_out,
                                       coordinate_space=getattr(args, "coordinate_space", "normalized_local"))
            except Exception as exc:
                print(f"SKIP {stem}: {exc}")
                skipped += 1
            done += 1
            if done % 100 == 0:
                print(f"[chart-ids] {done} charts processed, {total} tiles", flush=True)
        manifest = {
            "category": CATEGORY,
            "mode": "chart_ids_file",
            "chart_ids_file": str(args.chart_ids_file),
            "charts_requested": len(ids),
            "charts_processed": done,
            "charts_skipped": skipped,
            "tiles_written": total,
            "output_dir": str(out_root),
        }
        with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(json.dumps(manifest, indent=2))
        return 0

    if args.image_dir and args.label_dir:
        image_dir = Path(args.image_dir)
        label_dir = Path(args.label_dir)
    else:
        image_dir, label_dir = resolve_source_dirs(root, args.source)

    if args.source == "train_1" and not (args.image_dir and args.label_dir):
        print(
            "WARNING: --source train_1 overlaps Phase B/C training data. "
            "Prefer --source testing (default)."
        )

    if args.source == "train_1":
        out_root = Path(args.output_dir or root / "stage2_train1")
        holdout_root = Path(args.holdout_dir or root / "stage2_train1_holdout")
    else:
        out_root = Path(args.output_dir or root / "stage2_v2_1")
        holdout_root = Path(args.holdout_dir or root / "stage2_v2_1_holdout")
    manifest_path = Path(args.split_manifest or root / "split_manifest.json")

    excluded: Set[str] = set()
    if args.source == "testing":
        excluded = load_phase_train_stems(root, manifest_path)
        print(f"Phase B/C train stems to exclude: {len(excluded)}")
    else:
        print("Source train_1: using Phase B/C training pool (no stem exclusion).")

    pairs = collect_pairs(image_dir, label_dir)
    if excluded:
        pairs = filter_unused_pairs(pairs, excluded)
    if not pairs:
        print(f"No unused image/label pairs under {image_dir} / {label_dir}")
        return 1

    print(f"Source pool after exclusion: {len(pairs)} KM charts ({image_dir})")

    rng = np.random.default_rng(args.seed)
    order = np.arange(len(pairs))
    rng.shuffle(order)
    pairs = [pairs[i] for i in order]

    if args.max_charts is not None:
        pairs = pairs[: args.max_charts]

    n_holdout = max(1, int(len(pairs) * args.holdout_fraction))
    holdout_pairs = pairs[:n_holdout]
    train_pairs = pairs[n_holdout:]

    if args.clear_output:
        print("Clearing prior tile outputs...")
        clear_tile_output(out_root)
        clear_tile_output(holdout_root)

    stats = {"train_tiles": 0, "holdout_tiles": 0, "charts_skipped": 0}

    def run_split(split_pairs: Sequence[Tuple[Path, Path]], base: Path, desc: str) -> int:
        img_out = base / "images" / CATEGORY
        lbl_out = base / "labels" / CATEGORY
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        total = 0
        for img_path, label_path in tqdm(split_pairs, desc=desc):
            try:
                n = process_chart(
                    img_path,
                    label_path,
                    img_out,
                    lbl_out,
                    coordinate_space=args.coordinate_space,
                )
                total += n
            except Exception as exc:
                print(f"SKIP {label_path.name}: {exc}")
                stats["charts_skipped"] += 1
        return total

    stats["train_tiles"] = run_split(train_pairs, out_root, "stage2-train")
    stats["holdout_tiles"] = run_split(holdout_pairs, holdout_root, "stage2-holdout")

    manifest = {
        "category": CATEGORY,
        "coordinate_space": args.coordinate_space,
        "coord_decimals": COORD_DECIMALS,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "max_points_per_tile": MAX_POINTS_PER_TILE,
        "max_censors_per_tile": MAX_CENSORS_PER_TILE,
        "source": args.source,
        "excluded_train_stems": len(excluded),
        "source_image_dir": str(image_dir),
        "source_label_dir": str(label_dir),
        "train_dir": str(out_root),
        "holdout_dir": str(holdout_root),
        "source_charts_train": len(train_pairs),
        "source_charts_holdout": len(holdout_pairs),
        **stats,
    }
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Wrote manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
