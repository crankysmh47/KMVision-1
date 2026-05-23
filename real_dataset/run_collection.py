"""
One-shot real-world data collection: PMC ID search, then bulk image download.

Usage:
    python real_dataset/run_collection.py
    python real_dataset/run_collection.py --km-only
    python real_dataset/run_collection.py --skip-scrape
    python real_dataset/run_collection.py --target-km 300
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(command: list[str]) -> None:
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PMC scrape + image extraction")
    parser.add_argument("--km-only", action="store_true", help="Collect Kaplan-Meier images only")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip PMC ID search and only download images")
    parser.add_argument("--fresh-ids", action="store_true", help="Rebuild PMC ID lists from scratch")
    parser.add_argument("--target-km", type=int, default=250)
    parser.add_argument("--target-forest", type=int, default=125)
    parser.add_argument("--target-wf", type=int, default=125)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    types = ["km"] if args.km_only else ["km", "forest", "wf"]
    python = sys.executable

    if not args.skip_scrape:
        scrape_cmd = [python, str(ROOT / "scraper.py"), "--types", *types]
        if args.fresh_ids:
            scrape_cmd.append("--fresh")
        run_step(scrape_cmd)

    extract_cmd = [
        python,
        str(ROOT / "extracter.py"),
        "--types",
        *types,
        "--target-km",
        str(args.target_km),
        "--target-forest",
        str(args.target_forest),
        "--target-wf",
        str(args.target_wf),
        "--delay",
        str(args.delay),
    ]
    run_step(extract_cmd)

    print("\nCollection complete.")
    print("Start labeling with:")
    print("  python real_dataset/labeler.py")


if __name__ == "__main__":
    main()
