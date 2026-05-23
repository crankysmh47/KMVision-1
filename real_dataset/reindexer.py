import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import PROGRESS_FILE, curated_count, images_dir, inbox_count  # noqa: E402


def update_json_counts() -> None:
    """Sync progress.json downloaded counts with files on disk."""
    if not PROGRESS_FILE.exists():
        return
    with open(PROGRESS_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    data["downloaded"] = {
        "km": inbox_count("km"),
        "forest": inbox_count("forest"),
        "wf": inbox_count("wf"),
    }
    data["curated"] = {
        "km": curated_count("km"),
        "forest": curated_count("forest"),
        "wf": curated_count("wf"),
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
    print(f"progress.json updated: {data['downloaded']}")
def reindex_directory(directory_path, chart_type):
    if not os.path.exists(directory_path):
        return 0

    # 1. Get ONLY the files that still exist in the folder
    files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
    
    # 2. Sort them numerically based on the number currently in the name
    # This ensures that chart_003 stays before chart_006 even if 004/005 are gone
    def extract_number(filename):
        match = re.search(r'chart_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_number)

    # --- THE TWO-PASS FIX ---
    
    # Pass 1: Rename everything to a .tmp extension
    # This "unregisters" the old names from Windows so we have a clean slate
    temp_list = []
    for filename in files:
        old_path = os.path.join(directory_path, filename)
        temp_name = filename + ".tmp"
        temp_path = os.path.join(directory_path, temp_name)
        os.rename(old_path, temp_path)
        temp_list.append(temp_name)

    # Pass 2: Assign brand new, sequential numbers (001, 002, 003, 004...)
    # This will turn [chart_001, chart_002, chart_003, chart_006] 
    # into [chart_001, chart_002, chart_003, chart_004]
    for i, temp_name in enumerate(temp_list, start=1):
        old_temp_path = os.path.join(directory_path, temp_name)
        new_name = f"chart_{i:03d}_{chart_type}.png"
        new_path = os.path.join(directory_path, new_name)
        
        os.rename(old_temp_path, new_path)

    count = len(temp_list)
    print(f"✅ {chart_type.upper()}: Closed gaps for {count} images.")
    return count

# --- Execution ---
directories = {
    "km": images_dir("km"),
    "forest": images_dir("forest"),
    "wf": images_dir("wf"),
}

final_counts = {}

for chart_type, path in directories.items():
    count = reindex_directory(str(path), chart_type)
    final_counts[chart_type] = count

update_json_counts()

print("\nAll gaps closed. Files are sequential (001, 002, 003, ...).")