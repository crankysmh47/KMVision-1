"""
Download figure images from PMC articles into the labeling inbox.

New images land in real_dataset/inbox/{km,forest,wf}/ so you can review them
in labeler.py before promoting accepted charts to images_{type}/.

Usage:
    python real_dataset/extracter.py
    python real_dataset/extracter.py --types km --target 250
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (  # noqa: E402
    PMC_ID_FILES,
    PROGRESS_FILE,
    TARGETS,
    curated_count,
    ensure_dirs,
    inbox_count,
    inbox_dir,
)

SKIP_URL_FRAGMENTS = ("logo", "icon", "google", "button", "avatar", "banner")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    return {"km": 0, "forest": 0, "wf": 0, "downloaded": {"km": 0, "forest": 0, "wf": 0}}


def save_progress(chart_type: str, index: int, downloaded: dict[str, int]) -> None:
    progress = load_progress()
    progress[chart_type] = index
    progress["downloaded"] = downloaded
    with open(PROGRESS_FILE, "w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=4)


def get_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Auto-download a ChromeDriver build matching the installed Chrome (e.g. 148.x).
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def restart_driver(driver: webdriver.Chrome | None) -> webdriver.Chrome:
    if driver is not None:
        try:
            driver.quit()
        except Exception:
            pass
    return get_driver()


def next_filename(chart_type: str, current_count: int, saved_in_article: int) -> str:
    file_id = current_count + saved_in_article + 1
    return f"raw_{file_id:04d}_{chart_type}.png"


def scrape_pmc_verified(
    pmc_id: str,
    chart_type: str,
    driver: webdriver.Chrome,
    session: requests.Session,
    current_count: int,
    seen_urls: set[str],
) -> int:
    target_dir = inbox_dir(chart_type)
    ensure_dirs(chart_type)

    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
    saved_in_article = 0

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "img")))
        except TimeoutException:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        for cookie in driver.get_cookies():
            session.cookies.set(cookie["name"], cookie["value"])

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
        time.sleep(1.5)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        img_elements = soup.select("figure img, .part-figure img, .fig img")

        for img in img_elements:
            src = img.get("data-src") or img.get("src")
            if not src or any(fragment in src.lower() for fragment in SKIP_URL_FRAGMENTS):
                continue

            full_url = urljoin(url, src)
            if full_url in seen_urls:
                continue

            try:
                img_response = session.get(full_url, timeout=10)
                content_type = img_response.headers.get("Content-Type", "")
                if "image" not in content_type:
                    continue

                image = Image.open(io.BytesIO(img_response.content))
                if image.width < 120 or image.height < 120:
                    continue

                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")

                filename = next_filename(chart_type, current_count, saved_in_article)
                image.save(target_dir / filename, "PNG")
                seen_urls.add(full_url)
                print(f"    saved: {filename}")
                saved_in_article += 1
            except Exception:
                continue

    except Exception as exc:
        print(f"    error on {pmc_id}: {exc}")

    return saved_in_article


def run_extraction(types: list[str], targets: dict[str, int], delay: float) -> None:
    progress = load_progress()
    driver = get_driver()
    session = requests.Session()
    seen_urls: set[str] = set()
    consecutive_driver_errors = 0

    try:
        for chart_type in types:
            id_file = PMC_ID_FILES[chart_type]
            if not id_file.exists():
                print(f"Skipping {chart_type}: missing {id_file}")
                continue

            with open(id_file, encoding="utf-8") as handle:
                ids = [line.strip() for line in handle if line.strip()]

            target_goal = targets[chart_type]
            current_total = inbox_count(chart_type)
            curated = curated_count(chart_type)
            start_idx = progress.get(chart_type, 0)

            if current_total >= target_goal:
                print(f"{chart_type.upper()} inbox already at {current_total}/{target_goal} (curated: {curated}, untouched)")
                continue

            print(
                f"\nCollecting {chart_type.upper()} into inbox/ from PMC index {start_idx} "
                f"(inbox {current_total}/{target_goal}, curated {curated} in images_{chart_type}/ — not touched)"
            )

            for i in range(start_idx, len(ids)):
                if current_total >= target_goal:
                    print(f"Target reached for {chart_type}.")
                    break

                pmc_id = ids[i]
                print(f"[{i + 1}/{len(ids)}] {pmc_id} ...")
                try:
                    new_saved = scrape_pmc_verified(
                        pmc_id, chart_type, driver, session, current_total, seen_urls
                    )
                    consecutive_driver_errors = 0
                except WebDriverException as exc:
                    consecutive_driver_errors += 1
                    print(f"    driver error on {pmc_id}: {exc}")
                    new_saved = 0
                    if consecutive_driver_errors >= 2:
                        print("    restarting ChromeDriver ...")
                        driver = restart_driver(driver)
                        session = requests.Session()
                        consecutive_driver_errors = 0

                current_total += new_saved

                downloaded = progress.get("downloaded", {"km": 0, "forest": 0, "wf": 0})
                downloaded[chart_type] = current_total
                save_progress(chart_type, i + 1, downloaded)
                time.sleep(delay)

    finally:
        driver.quit()
        print(f"\nSession finished. Progress saved to {PROGRESS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk download PMC figure images")
    parser.add_argument("--types", nargs="+", default=["km", "forest", "wf"], choices=["km", "forest", "wf"])
    parser.add_argument("--target-km", type=int, default=TARGETS["km"])
    parser.add_argument("--target-forest", type=int, default=TARGETS["forest"])
    parser.add_argument("--target-wf", type=int, default=TARGETS["wf"])
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between PMC articles")
    args = parser.parse_args()

    targets = {
        "km": args.target_km,
        "forest": args.target_forest,
        "wf": args.target_wf,
    }
    run_extraction(args.types, targets, args.delay)


if __name__ == "__main__":
    main()
