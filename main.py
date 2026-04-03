import os
import argparse
import concurrent.futures
from tqdm import tqdm
import random

from generate_km import generate_km_chart
from generate_clinical import generate_forest_plot, generate_waterfall_plot
from generate_anchor import generate_bar_chart, generate_line_chart, generate_scatter_chart
from augment import augment_images

def worker(task_idx):
    try:
        r = random.random()
        if r < 0.50:
            generate_km_chart()
        elif r < 0.60:
            generate_forest_plot()
        elif r < 0.70:
            generate_waterfall_plot()
        elif r < 0.80:
            generate_bar_chart()
        elif r < 0.90:
            generate_line_chart()
        else:
            generate_scatter_chart()
        return True
    except Exception:
        # Prevent random unhandled matplotlib errors from killing the process pool
        return False

def main():
    parser = argparse.ArgumentParser(description="Synthetic Clinical Chart Generation Pipeline")
    parser.add_argument("--num_samples", type=int, default=100, help="Total number of charts to generate")
    args = parser.parse_args()

    num_samples = args.num_samples
    print(f"Generating {num_samples} synthetic charts using {os.cpu_count()} cores...")

    os.makedirs(r"C:\sem4\KMVision-1 Data\dataset\images", exist_ok=True)
    os.makedirs(r"C:\sem4\KMVision-1 Data\dataset\labels", exist_ok=True)

    import multiprocessing as mp
    
    # Use multiprocessing avoiding the GIL for heavily CPU-bound matplotlib ops
    # maxtasksperchild=1000 CRITICAL: Forces each worker to restart after 1000 charts, 
    # completely wiping any un-collectable memory or Windows OS GDI object leaks.
    with mp.Pool(processes=os.cpu_count(), maxtasksperchild=1000) as pool:
        # Use chunksize to reduce IPC overhead on Windows
        chunksize = max(1, num_samples // (os.cpu_count() * 4))
        if chunksize > 50: chunksize = 50
        
        list(tqdm(pool.imap_unordered(worker, range(num_samples), chunksize=chunksize), total=num_samples))
        
    print(f"Completed generation of {num_samples} charts.")
    
    print("Applying adversarial augmentations to 10% of generated images...")
    augment_images(r"C:\sem4\KMVision-1 Data\dataset\images", ratio=0.1)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    # Required wrap for Windows multiprocessing forks
    main()
