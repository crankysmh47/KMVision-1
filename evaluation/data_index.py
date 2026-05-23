"""
Dataset indexing — mirrors train_phase_b.ClinicalChartDataset sample selection.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Sequence, Tuple

Sample = Tuple[str, str]  # (image_path, label_path)

# Matches train_phase_b.py SUBSET_SIZE
PHASE_B_MAX_SAMPLES = 100_000
DEFAULT_SPLIT_SEED = 42


def collect_all_valid_samples(image_dir: str, label_dir: str) -> Dict[str, List[Sample]]:
    """Walk labels/ and pair each .json with a .png/.jpg in images/{category}/."""
    category_samples: Dict[str, List[Sample]] = {}

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Missing image_dir or label_dir: {image_dir}, {label_dir}")

    for root, _, files in os.walk(label_dir):
        category = os.path.basename(root)
        if category == os.path.basename(label_dir):
            continue

        if category not in category_samples:
            category_samples[category] = []

        for label_file in files:
            if not label_file.endswith(".json"):
                continue
            base_name = os.path.splitext(label_file)[0]
            img_path = os.path.join(image_dir, category, f"{base_name}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(image_dir, category, f"{base_name}.jpg")
                if not os.path.exists(img_path):
                    continue
            category_samples[category].append((img_path, os.path.join(root, label_file)))

    return category_samples


def build_phase_b_training_subset(
    image_dir: str,
    label_dir: str,
    *,
    max_samples: int = PHASE_B_MAX_SAMPLES,
    seed: int = DEFAULT_SPLIT_SEED,
) -> List[Sample]:
    """
    Reproduce Phase B balanced sampling (train_phase_b.ClinicalChartDataset).

    Note: train_phase_b.py does not call random.seed(), so an actual training run
    may have drawn a different 100k subset. This uses `seed` for a reproducible split.
    """
    category_samples = collect_all_valid_samples(image_dir, label_dir)
    num_categories = len(category_samples)
    if num_categories == 0:
        raise ValueError(f"No valid label categories in {label_dir}")

    rng = random.Random(seed)
    samples_per_category = max_samples // num_categories
    final_samples: List[Sample] = []

    for cat, samples in category_samples.items():
        if len(samples) >= samples_per_category:
            final_samples.extend(rng.sample(samples, samples_per_category))
        else:
            final_samples.extend(samples)

    rng.shuffle(final_samples)
    return final_samples[:max_samples]


def split_train_and_test(
    image_dir: str,
    label_dir: str,
    *,
    max_train_samples: int = PHASE_B_MAX_SAMPLES,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[Sample], List[Sample]]:
    """Training subset vs all remaining valid pairs."""
    category_samples = collect_all_valid_samples(image_dir, label_dir)
    train = build_phase_b_training_subset(
        image_dir, label_dir, max_samples=max_train_samples, seed=seed
    )
    train_set = set(train)
    test: List[Sample] = []
    for samples in category_samples.values():
        for pair in samples:
            if pair not in train_set:
                test.append(pair)
    return train, test


def save_manifest(path: str, train: Sequence[Sample], test: Sequence[Sample], meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        **meta,
        "train_count": len(train),
        "test_count": len(test),
        "train": [{"image": i, "label": l} for i, l in train],
        "test": [{"image": i, "label": l} for i, l in test],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_manifest_train_pairs(manifest_path: str) -> List[Sample]:
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return [(e["image"], e["label"]) for e in data["train"]]
