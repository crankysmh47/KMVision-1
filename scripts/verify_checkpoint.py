"""Verify a training run wrote a complete checkpoint directory."""

from __future__ import annotations

import argparse
import os
import sys


def verify(checkpoint_dir: str) -> list[str]:
    errors = []
    if not os.path.isdir(checkpoint_dir):
        return [f"directory missing: {checkpoint_dir}"]
    for name in ("adapter_model.safetensors", "projector_weights.pth", "adapter_config.json"):
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            errors.append(f"missing file: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="e.g. checkpoints/phase_c_run1_minified/final")
    args = parser.parse_args()
    errors = verify(args.checkpoint_dir)
    if errors:
        print("CHECKPOINT INVALID:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"CHECKPOINT OK: {args.checkpoint_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
