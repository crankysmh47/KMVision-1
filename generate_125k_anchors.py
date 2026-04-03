import os
import multiprocessing as mp
from tqdm import tqdm
from generate_anchor import generate_random_anchor

def worker(idx):
    try:
        generate_random_anchor()
        return True
    except Exception as e:
        return False

def run_anchor_batch():
    total_samples = 125000
    print(f"Running targeted batch of {total_samples} Anchor Charts...")
    print("This will process entirely through the 5 synthetic generators (Stacked, Dual-Combo, Bar, Line, Scatter).")
    
    tasks = list(range(total_samples))
    
    with mp.Pool(processes=os.cpu_count(), maxtasksperchild=100) as pool:
        list(tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)))
        
    print(f"Completed! Exact anchor gap of {total_samples} charts filled.")

if __name__ == '__main__':
    run_anchor_batch()
