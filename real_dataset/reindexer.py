import os
import re
import json

PROGRESS_FILE = "real_dataset/progress.json"

def update_json_counts(counts):
    """Updates the global_total in progress.json to match actual files on disk."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
        
        # Sync the total to the sum of what actually exists in folders now
        data["global_total"] = sum(counts.values())
        
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"📊 progress.json updated. New global total: {data['global_total']}")

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
    "km": "real_dataset/images_km",
    "forest": "real_dataset/images_forest",
    "wf": "real_dataset/images_wf"
}

final_counts = {}

for c_type, path in directories.items():
    count = reindex_directory(path, c_type)
    final_counts[c_type] = count

update_json_counts(final_counts)


print("\n✨ All gaps closed. Your files are now perfectly sequential (001, 002, 003, 004...).")