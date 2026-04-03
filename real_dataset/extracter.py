import requests
import os
import time
import sys

# --- Configuration & Debug Settings ---
TARGET_IMAGES = 500
IMAGE_DIR = "real_dataset/images"
LOG_FILE = "real_dataset/extraction_debug_log.txt"
os.makedirs(IMAGE_DIR, exist_ok=True)

def log_debug(message):
    """Writes to a log file and prints to terminal for real-time tracking."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def download_high_res_km(pmc_id, global_count):
    """
    Directly targets the PMC /bin/ server where high-res figures are stored.
    Bypasses HTML/Scraping restrictions.
    """
    # Note: PMC storage often uses the raw ID (without 'PMC' prefix) for paths
    clean_id = pmc_id.replace("PMC", "")
    base_bin_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{clean_id}/bin/"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) MedicalDataCollector/1.1'}
    saved_in_article = 0

    # Journals usually have 1-6 figures. We probe for common naming conventions.
    # Patterns: F1.jpg (Standard), fig1.jpg (Alternative), F1.png (High-Res)
    for fig_num in range(1, 7):
        if global_count + saved_in_article >= TARGET_IMAGES:
            break

        found_fig = False
        for ext in ["jpg", "png", "jpeg"]:
            for prefix in [f"F{fig_num}", f"fig{fig_num}"]:
                img_url = f"{base_bin_url}{prefix}.{ext}"
                
                try:
                    # Use a HEAD request to check existence without wasting bandwidth
                    response = requests.head(img_url, headers=headers, timeout=5)
                    
                    if response.status_code == 200:
                        # File exists! Now GET the data.
                        img_data = requests.get(img_url, headers=headers).content
                        
                        current_id = global_count + saved_in_article + 1
                        file_name = f"chart_{current_id:03d}_km.png"
                        file_path = os.path.join(IMAGE_DIR, file_name)
                        
                        with open(file_path, 'wb') as f:
                            f.write(img_data)
                        
                        log_debug(f"SUCCESS: {pmc_id} -> Saved {prefix}.{ext} as {file_name}")
                        saved_in_article += 1
                        found_fig = True
                        break # Found this figure, move to next fig_num
                except Exception as e:
                    continue
            if found_fig: break # Stop checking extensions for this figure number

    return saved_in_article

# --- Main Execution Loop ---
if __name__ == "__main__":
    total_images_saved = 0
    
    # Load IDs
    try:
        with open("real_dataset/pmc_ids.txt", "r") as f:
            pmc_ids = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("❌ Error: pmc_ids.txt not found. Please run your ID collector script first.")
        sys.exit()

    log_debug(f"STARTING EXTRACTION: Goal = {TARGET_IMAGES} images.")

    for index, pmc_id in enumerate(pmc_ids, start=1):
        if total_images_saved >= TARGET_IMAGES:
            break
        
        # Periodic "Milestone" Debugging
        if index % 10 == 0:
            log_debug(f"PROGRESS: Processed {index} articles. Total Images: {total_images_saved}/{TARGET_IMAGES}")

        new_count = download_high_res_km(pmc_id, total_images_saved)
        total_images_saved += new_count
        
        if new_count == 0:
            # Silent debug for skipped articles
            with open(LOG_FILE, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] SKIP: {pmc_id} (No KM-style figures found in bin)\n")

        # Rate Limiting: Stay under the radar
        time.sleep(1.5)

    log_debug(f"FINISHED: Total Images Extracted: {total_images_saved}")
    if total_images_saved < TARGET_IMAGES:
        log_debug("⚠️ WARNING: Ran out of PMC IDs before hitting 500 images. Collect more IDs!")