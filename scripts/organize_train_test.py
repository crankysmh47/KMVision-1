"""
Move Phase-B training subset into train_1/ and holdout into testing/.

Usage (from repo root):
  python scripts/organize_train_test.py
  python scripts/organize_train_test.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from evaluation.data_index import (
    DEFAULT_SPLIT_SEED,
    PHASE_B_MAX_SAMPLES,
    save_manifest,
    split_train_and_test,
)

DEFAULT_DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"


def _relocate_pair(
    img_path: str,
    label_path: str,
    src_image_root: str,
    src_label_root: str,
    dst_image_root: str,
    dst_label_root: str,
    *,
    dry_run: bool,
) -> None:
    rel_img = os.path.relpath(img_path, src_image_root)
    rel_lbl = os.path.relpath(label_path, src_label_root)
    dst_img = os.path.join(dst_image_root, rel_img)
    dst_lbl = os.path.join(dst_label_root, rel_lbl)
    if dry_run:
        return
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
    if os.path.abspath(img_path) != os.path.abspath(dst_img):
        if os.path.exists(dst_img):
            os.remove(dst_img)
        shutil.move(img_path, dst_img)
    if os.path.abspath(label_path) != os.path.abspath(dst_lbl):
        if os.path.exists(dst_lbl):
            os.remove(dst_lbl)
        shutil.move(label_path, dst_lbl)


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize dataset into train_1 and testing folders.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--max-train", type=int, default=PHASE_B_MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_images = os.path.join(args.dataset_root, "images")
    src_labels = os.path.join(args.dataset_root, "labels")
    train_root = os.path.join(args.dataset_root, "train_1")
    test_root = os.path.join(args.dataset_root, "testing")

    print("Indexing dataset (Phase B selection logic)...")
    train, test = split_train_and_test(
        src_images, src_labels, max_train_samples=args.max_train, seed=args.seed
    )
    print(f"Train: {len(train):,}  |  Test: {len(test):,}")

    manifest_path = os.path.join(args.dataset_root, "split_manifest.json")
    save_manifest(
        manifest_path,
        train,
        test,
        meta={
            "selection": "train_phase_b.ClinicalChartDataset balanced sampling",
            "max_train_samples": args.max_train,
            "seed": args.seed,
            "note": "Original Phase B training did not set random.seed(); this split is reproducible via seed.",
            "train_1_dir": train_root,
            "testing_dir": test_root,
        },
    )
    print(f"Wrote manifest: {manifest_path}")

    if args.dry_run:
        print("Dry run — no files moved.")
        return

    train_img_root = os.path.join(train_root, "images")
    train_lbl_root = os.path.join(train_root, "labels")
    test_img_root = os.path.join(test_root, "images")
    test_lbl_root = os.path.join(test_root, "labels")

    for pairs, dst_img, dst_lbl, name in [
        (train, train_img_root, train_lbl_root, "train_1"),
        (test, test_img_root, test_lbl_root, "testing"),
    ]:
        print(f"Moving {len(pairs):,} samples -> {name}/")
        for img_path, label_path in tqdm(pairs, desc=name):
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue
            _relocate_pair(
                img_path,
                label_path,
                src_images,
                src_labels,
                dst_img,
                dst_lbl,
                dry_run=False,
            )

    print("Done. Original images/ and labels/ should now be empty or sparse.")


if __name__ == "__main__":
    main()
