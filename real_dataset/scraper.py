"""
Fetch PMC article IDs from NCBI E-utilities for bulk image collection.

Usage:
    python real_dataset/scraper.py
    python real_dataset/scraper.py --fresh
    python real_dataset/scraper.py --types km forest wf
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import PMC_ID_FILES  # noqa: E402

SEARCHES = {
    "km": [
        "Kaplan-Meier Survival",
        "Survival Analysis Kaplan Meier",
    ],
    "forest": [
        "Forest Plot",
        "Forest Plot meta analysis",
    ],
    "wf": [
        "Waterfall Plot",
        "Waterfall plot oncology",
    ],
}


def load_existing_ids(filename: str) -> set[str]:
    if not os.path.exists(filename):
        return set()
    with open(filename, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def get_and_save_plos_ids(query: str, filename: str, count: int = 1000, email: str = "your_email@example.com") -> int:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_term = f'"{query}" AND "PLOS ONE"[Journal]'
    params = {
        "db": "pmc",
        "term": search_term,
        "retmax": count,
        "retmode": "xml",
        "tool": "MedicalDataCollector",
        "email": email,
    }

    print(f"Searching PMC for: {search_term}")

    try:
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"API error: {response.status_code}")
            return 0

        root = ET.fromstring(response.content)
        found_count = root.find("Count").text if root.find("Count") is not None else "0"
        print(f"NCBI found {found_count} total matches.")

        ids = [id_tag.text for id_tag in root.findall(".//Id")]
        if not ids:
            print(f"No IDs returned for '{query}'.")
            return 0

        existing = load_existing_ids(filename)
        new_ids = [f"PMC{pmcid}" for pmcid in ids if f"PMC{pmcid}" not in existing]
        if not new_ids:
            print(f"All {len(ids)} IDs already present in {filename}")
            return 0

        with open(filename, "a", encoding="utf-8") as handle:
            for pmcid in new_ids:
                handle.write(f"{pmcid}\n")

        print(f"Added {len(new_ids)} new IDs to {filename} ({len(existing) + len(new_ids)} total)")
        return len(new_ids)
    except Exception as exc:
        print(f"Connection error: {exc}")
        return 0


def scrape_types(types: list[str], fresh: bool, email: str) -> None:
    for chart_type in types:
        filename = str(PMC_ID_FILES[chart_type])
        if fresh and os.path.exists(filename):
            os.remove(filename)
            print(f"Cleared {filename}")

        for query in SEARCHES[chart_type]:
            get_and_save_plos_ids(query, filename, count=1000, email=email)
            time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PMC IDs for real-world chart images")
    parser.add_argument("--types", nargs="+", default=["km", "forest", "wf"], choices=["km", "forest", "wf"])
    parser.add_argument("--fresh", action="store_true", help="Clear existing ID lists before searching")
    parser.add_argument("--email", default="your_email@example.com", help="Contact email for NCBI E-utilities")
    args = parser.parse_args()
    scrape_types(args.types, args.fresh, args.email)


if __name__ == "__main__":
    main()
