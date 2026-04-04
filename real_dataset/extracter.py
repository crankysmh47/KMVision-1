import os
import time
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
from PIL import Image
import io

# --- Persistence Layer ---
PROGRESS_FILE = "real_dataset/progress.json"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"km": 0, "forest": 0, "wf": 0, "global_total": 0}

def save_progress(chart_type, index, global_total):
    progress = load_progress()
    progress[chart_type] = index
    progress["global_total"] = global_total
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=4)

def get_driver():
    options = Options()
    options.add_argument("--headless") 
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    return webdriver.Chrome(options=options)

def scrape_pmc_verified(pmc_id, chart_type, driver, session, current_global_count):
    target_dir = f"real_dataset/images_{chart_type}"
    os.makedirs(target_dir, exist_ok=True)
    
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
    saved_in_article = 0

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "img")))

        for cookie in driver.get_cookies():
            session.cookies.set(cookie['name'], cookie['value'])
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(1)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        img_elements = soup.select('figure img, .part-figure img, .fig img')

        for img in img_elements:
            src = img.get('data-src') or img.get('src')
            if not src or any(x in src.lower() for x in ["logo", "icon", "google", "button"]):
                continue
            
            full_url = urljoin(url, src)
            
            try:
                img_response = session.get(full_url, timeout=10)
                if "image" in img_response.headers.get('Content-Type', ''):
                    image = Image.open(io.BytesIO(img_response.content))
                    
                    if image.mode in ("RGBA", "P"):
                        image = image.convert("RGB")
                    
                    # --- NEW NAMING CONVENTION ---
                    # chart_001_km.png, chart_002_km.png...
                    file_id = current_global_count + saved_in_article + 1
                    filename = f"chart_{file_id:03d}_{chart_type}.png"
                    
                    image.save(os.path.join(target_dir, filename), "PNG")
                    print(f"    ✅ Saved: {filename}")
                    saved_in_article += 1
            except Exception:
                continue

    except Exception as e:
        print(f"    ❌ Error on {pmc_id}: {e}")
    
    return saved_in_article

# --- Main Execution ---
TARGETS = {"km": 250, "forest": 125, "wf": 125}
FILES = {
    "km": "real_dataset/plos_id_km.txt",
    "forest": "real_dataset/plos_id_forest.txt",
    "wf": "real_dataset/plos_id_wf.txt"
}

progress = load_progress()
driver = get_driver()
session = requests.Session()

try:
    for c_type, target_goal in TARGETS.items():
        if not os.path.exists(FILES[c_type]): continue
        
        with open(FILES[c_type], "r") as f:
            ids = [line.strip() for line in f.readlines()]
        
        # Determine starting point for this specific list
        start_idx = progress.get(c_type, 0)
        
        # Count how many images we ALREADY have in this folder
        target_dir = f"real_dataset/images_{c_type}"
        os.makedirs(target_dir, exist_ok=True)
        current_type_total = len([n for n in os.listdir(target_dir) if n.endswith('.png')])

        if current_type_total >= target_goal:
            print(f"🏆 {c_type.upper()} already complete ({current_type_total}/{target_goal}).")
            continue

        print(f"\n📂 RESUMING BATCH: {c_type.upper()} from ID index {start_idx}")
        
        for i in range(start_idx, len(ids)):
            if current_type_total >= target_goal:
                print(f"🎯 Target reached for {c_type}.")
                break
            
            pmc_id = ids[i]
            print(f"[{i}/{len(ids)}] Probing {pmc_id}...")
            
            # Pass the current type total to maintain naming
            new_saved = scrape_pmc_verified(pmc_id, c_type, driver, session, current_type_total)
            
            current_type_total += new_saved
            
            # Save progress after every PMC ID processed
            save_progress(c_type, i + 1, sum(TARGETS.values())) 
            
            time.sleep(2)

finally:
    driver.quit()
    print(f"\n✨ Session Paused/Finished. Progress saved to {PROGRESS_FILE}")