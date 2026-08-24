"""V2-SFT: continue the Phase C Run 2 macro model on the extended-schema corpus.

Thin wrapper over train_phase_c with v2 defaults:
  images  : {root}/train_v2/images
  labels  : {root}/labels_compressed_v2
  init    : checkpoints/phase_c_run2_chatml/final   (data-only variable)
  seq_len : 1024 (v2 targets carry title/unit/HR/CI/p/risk-table)
  ids     : {root}/train_v2_train_ids.txt (excludes val_v2 + frozen_test_v2)

Usage:
  venv/Scripts/python.exe train_v2_sft.py --max_global_steps 700 --auto_resume
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"


def main() -> None:
    sys.argv = [
        "train_phase_c.py",
        "--dataset_root", DATASET_ROOT,
        "--image_dir", os.path.join(DATASET_ROOT, "train_v2", "images"),
        "--label_dir", os.path.join(DATASET_ROOT, "labels_compressed_v2"),
        "--init_checkpoint", "checkpoints/phase_c_run2_chatml/final",
        "--output_dir", "checkpoints/v2_sft_run1",
        "--use_chatml",
        "--seq_len", "2560",
        "--train_ids_file", os.path.join(DATASET_ROOT, "train_v2_train_ids.txt"),
        "--subset_size", "11000",
        "--max_global_steps", "700",
        "--checkpoint_every", "25",
        "--learning_rate", "3e-5",
        "--auto_resume",
    ]
    from train_phase_c import main as phase_c_main

    phase_c_main()


if __name__ == "__main__":
    main()
