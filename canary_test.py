import concurrent.futures
from tqdm import tqdm
import os
import multiprocessing as mp

from generate_km import generate_km_chart
from generate_clinical import generate_forest_plot, generate_waterfall_plot
from generate_anchor import generate_bar_chart, generate_line_chart, generate_scatter_chart

def canary_worker(task_info):
    task_type, idx = task_info
    try:
        if task_type == 'km':
            generate_km_chart()
        elif task_type == 'forest':
            generate_forest_plot()
        elif task_type == 'waterfall':
            generate_waterfall_plot()
        elif task_type == 'bar':
            generate_bar_chart()
        elif task_type == 'line':
            generate_line_chart()
        elif task_type == 'scatter':
            generate_scatter_chart()
        return True
    except Exception as e:
        print(f"Failed {task_type} chart: {e}")
        return False

def run_canary():
    output_images = r"C:\sem4\KMVision-1 Data\dataset\images"
    output_labels = r"C:\sem4\KMVision-1 Data\dataset\labels"
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    print("Purging previous dataset for clean canary run...")
    for f in os.listdir(output_images):
        os.remove(os.path.join(output_images, f))
    for f in os.listdir(output_labels):
        os.remove(os.path.join(output_labels, f))

    print("Building explicitly balanced canary list: 250 KM, 100 Clinical, 150 Anchor...")
    tasks = (
        [("km", i) for i in range(250)] +
        [("forest", i) for i in range(50)] +
        [("waterfall", i) for i in range(50)] +
        [("bar", i) for i in range(50)] +
        [("line", i) for i in range(50)] +
        [("scatter", i) for i in range(50)]
    )

    print("Running Canary Batch across process pool...")
    with mp.Pool(processes=os.cpu_count(), maxtasksperchild=50) as pool:
        list(tqdm(pool.imap_unordered(canary_worker, tasks), total=len(tasks)))

    print("Canary generation complete!")

if __name__ == '__main__':
    run_canary()
