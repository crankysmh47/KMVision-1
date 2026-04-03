import os
from tqdm import tqdm
import multiprocessing as mp

from generate_anchor import generate_random_anchor

def canary_worker(idx):
    try:
        generate_random_anchor()
        return True
    except Exception as e:
        print(f"Failed anchor {idx}: {e}")
        return False

def run_canary_v2():
    print("Running Pure Synthetic Anchor verification batch (50 samples)...")
    tasks = list(range(50))

    with mp.Pool(processes=min(8, os.cpu_count()), maxtasksperchild=10) as pool:
        list(tqdm(pool.imap_unordered(canary_worker, tasks), total=len(tasks)))

    print("Canary v2 generation complete!")

if __name__ == '__main__':
    run_canary_v2()
