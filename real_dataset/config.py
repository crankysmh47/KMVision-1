"""Shared paths and helpers for the real-world dataset pipeline."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TARGETS = {"km": 250, "forest": 125, "wf": 125}

PMC_ID_FILES = {
    "km": ROOT / "plos_id_km.txt",
    "forest": ROOT / "plos_id_forest.txt",
    "wf": ROOT / "plos_id_wf.txt",
}

PROGRESS_FILE = ROOT / "progress.json"
LABELING_STATE_FILE = ROOT / "labeling_state.json"


def images_dir(chart_type: str) -> Path:
    return ROOT / f"images_{chart_type}"


def inbox_dir(chart_type: str) -> Path:
    return ROOT / "inbox" / chart_type


def discarded_dir(chart_type: str) -> Path:
    return ROOT / "discarded" / chart_type


def labels_dir(chart_type: str) -> Path:
    return ROOT / "labels" / chart_type


def ensure_dirs(chart_type: str) -> None:
    for path in (images_dir(chart_type), inbox_dir(chart_type), discarded_dir(chart_type), labels_dir(chart_type)):
        path.mkdir(parents=True, exist_ok=True)


def chart_number(filename: str) -> int:
    match = re.search(r"chart_(\d+)", filename)
    return int(match.group(1)) if match else 0


def raw_number(filename: str) -> int:
    match = re.search(r"raw_(\d+)", filename)
    return int(match.group(1)) if match else 0


def image_sort_key(path: Path) -> tuple[int, int]:
    """Sort curated chart_* before inbox raw_* files."""
    if path.name.startswith("raw_"):
        return (1, raw_number(path.name))
    return (0, chart_number(path.name))


def list_pngs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [directory / name for name in os.listdir(directory) if name.lower().endswith(".png")],
        key=image_sort_key,
    )


def flatten_nested_images(chart_type: str = "km") -> int:
    """
    Move PNGs from accidental nested folders (e.g. images_km/images_km/) into
    the canonical images_{type} directory without overwriting existing files.
    """
    ensure_dirs(chart_type)
    destination = images_dir(chart_type)
    moved = 0

    for nested in ROOT.rglob("*.png"):
        if nested.parent == destination:
            continue
        if chart_type not in nested.name:
            continue
        if not str(nested).startswith(str(ROOT)):
            continue
        if nested.parent == inbox_dir(chart_type) or nested.parent == discarded_dir(chart_type):
            continue

        target = destination / nested.name
        if target.exists():
            continue

        shutil.move(str(nested), str(target))
        moved += 1

    _remove_empty_dirs(ROOT / f"images_{chart_type}")
    return moved


def _remove_empty_dirs(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()


def label_path_for_image(image_path: Path, chart_type: str) -> Path:
    return labels_dir(chart_type) / f"{image_path.stem}.json"


def unlabeled_queue(chart_type: str) -> list[Path]:
    """Images in inbox or accepted folders that do not yet have a label file."""
    ensure_dirs(chart_type)
    seen: set[str] = set()
    queue: list[Path] = []

    # Curated images first (your existing 128), then new inbox scrapes.
    for folder in (images_dir(chart_type), inbox_dir(chart_type)):
        for image_path in list_pngs(folder):
            if image_path.name in seen:
                continue
            if label_path_for_image(image_path, chart_type).exists():
                continue
            seen.add(image_path.name)
            queue.append(image_path)

    return sorted(queue, key=image_sort_key)


def inbox_count(chart_type: str) -> int:
    """Raw scraped images waiting for review — does not include curated images_{type}/."""
    ensure_dirs(chart_type)
    return len(list_pngs(inbox_dir(chart_type)))


def curated_count(chart_type: str) -> int:
    """Accepted images kept after manual filtering/labeling."""
    ensure_dirs(chart_type)
    return len(list_pngs(images_dir(chart_type)))


def next_inbox_name(chart_type: str) -> str:
    existing = list_pngs(inbox_dir(chart_type))
    next_id = max((raw_number(path.name) for path in existing), default=0) + 1
    return f"raw_{next_id:04d}_{chart_type}.png"


def next_curated_name(chart_type: str) -> str:
    existing = list_pngs(images_dir(chart_type))
    next_id = max((chart_number(path.name) for path in existing), default=0) + 1
    return f"chart_{next_id:03d}_{chart_type}.png"


def total_downloaded(chart_type: str) -> int:
    """Backward-compatible alias — scrape progress tracks inbox only."""
    return inbox_count(chart_type)
