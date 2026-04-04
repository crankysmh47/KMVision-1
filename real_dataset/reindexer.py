import os
import re
import json

PROGRESS_FILE = "real_dataset/progress.json"

def update_json_counts(counts):
    """Updates the global_total in progress.json based on actual file counts."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
        
        # We don't change the 'index' (where we are in the PMC list), 
        # but we update the global_total to reflect what's actually on disk.
        data["global_total"] = sum(counts.values())
        
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"📊 progress.json updated. New global total: {data['global_total']}")

def reindex_directory(directory_path, chart_type):
    if not os.path.exists(directory_path):
        return 0

    files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
    
    # Sort numerically to keep the order you saw while deleting
    def extract_number(filename):
        match = re.search(r'chart_(\d+)', filename)
        return int(match.group(1)) if match else 0

    files.sort(key=extract_number)

    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(directory_path, filename)
        new_name = f"chart_{i:03d}_{chart_type}.png"
        new_path = os.path.join(directory_path, new_name)
        
        if old_path != new_path:
            os.rename(old_path, new_path)

    print(f"✅ {chart_type.upper()}: Cleaned and indexed {len(files)} images.")
    return len(files)

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

# Sync the JSON so the scraper knows how many more it needs to find
update_json_counts(final_counts)

print("\n✨ Workspace is now synchronized. You can safely run your scraper again to fill the gaps!")