import os
import time
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

def get_driver():
    options = Options()
    options.add_argument("--headless") 
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    return webdriver.Chrome(options=options)

def scrape_pmc_verified(pmc_id, chart_type, driver, session):
    # --- FIX: Dynamic Directory Selection ---
    target_dir = f"real_dataset/images_{chart_type}"
    os.makedirs(target_dir, exist_ok=True)
    
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
    saved_in_article = 0

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "img")))

        # Sync cookies to bypass 403 blocks
        for cookie in driver.get_cookies():
            session.cookies.set(cookie['name'], cookie['value'])
        
        # Scroll to ensure lazy-loaded images are triggered
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(1)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # Filter for actual figures to avoid site logos and icons
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
                    
                    # Get current count in THIS specific folder for naming
                    existing_files = len(os.listdir(target_dir))
                    filename = f"chart_{existing_files + 1}.png"
                    
                    image.save(os.path.join(target_dir, filename), "PNG")
                    print(f"   ✅ Saved to {chart_type}: {filename}")
                    saved_in_article += 1
            except Exception:
                continue

    except Exception as e:
        print(f"   ❌ Error on {pmc_id}: {e}")
    
    return saved_in_article

# --- Main Execution ---
# Define targets for each folder
TARGETS = {
    "km": 250,      # Folder: images_km
    "forest": 125,  # Folder: images_forest
    "wf": 125       # Folder: images_wf
}

FILES = {
    "km": "real_dataset/plos_id_km.txt",
    "forest": "real_dataset/plos_id_forest.txt",
    "wf": "real_dataset/plos_id_wf.txt"
}

driver = get_driver()
session = requests.Session()

for c_type, target_goal in TARGETS.items():
    if not os.path.exists(FILES[c_type]):
        print(f"Skipping {c_type} - ID file not found.")
        continue
    
    with open(FILES[c_type], "r") as f:
        # Deduplicate IDs to ensure unique article probing
        ids = list(set([line.strip() for line in f.readlines()]))
    
    print(f"\n📂 STARTING BATCH: {c_type.upper()} (Target: {target_goal})")
    
    current_type_total = 0
    for pmc_id in ids:
        if current_type_total >= target_goal:
            break
        
        count = scrape_pmc_verified(pmc_id, c_type, driver, session)
        current_type_total += count
        
        # Polite delay to prevent NIH server flags
        time.sleep(2)

driver.quit()
print(f"\n✨ DONE! Check the 'real_dataset/' folder for your three new directories.")