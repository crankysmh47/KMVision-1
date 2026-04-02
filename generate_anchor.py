import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, uuid, random
import numpy as np
import seaborn as sns
from schemas import AnchorChartSchema, AnchorSeries

OUTPUT_DIR = "dataset"
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

DEFAULT_FONT = 'DejaVu Sans'

def init_plot(basename):
    seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(seed)
    random.seed(seed)
    plt.rcParams.update({'font.family': DEFAULT_FONT})
    fig_width = random.uniform(6, 12)
    fig_height = random.uniform(5, 10)
    dpi = random.choice([100, 150, 200, 300]) 
    if fig_width * dpi > 1024: dpi = int(1024 / fig_width)
    if fig_height * dpi > 1024: dpi = int(1024 / fig_height)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    return fig, ax

def save_and_close(fig, output_basename, schema):
    img_path = os.path.join(OUTPUT_DIR, "images", f"{output_basename}.png")
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    plt.close('all')
    import gc
    gc.collect()
    with open(os.path.join(OUTPUT_DIR, "labels", f"{output_basename}.json"), 'w') as f:
        try: f.write(schema.model_dump_json(indent=2))
        except: f.write(schema.json(indent=2))

def generate_bar_chart(output_basename=None):
    if output_basename is None: output_basename = f"chart_{uuid.uuid4().hex[:8]}_bar"
    fig, ax = init_plot(output_basename)
    n_categories = random.randint(3, 8)
    categories = [f"Cat {chr(65+i)}" for i in range(n_categories)]
    values = np.random.uniform(10, 100, n_categories)
    sns.barplot(x=categories, y=values, ax=ax, hue=categories, palette="viridis", legend=False)
    ax.set_ylabel("Count")
    series_data = [(cat, float(val)) for cat, val in zip(categories, values)]
    schema = AnchorChartSchema(chart_type="bar", axes={"x": {"label": ""}, "y": {"label": "Count"}}, 
                               series=[AnchorSeries(label="Counts", data_points=series_data)])
    save_and_close(fig, output_basename, schema)

def generate_line_chart(output_basename=None):
    if output_basename is None: output_basename = f"chart_{uuid.uuid4().hex[:8]}_line"
    fig, ax = init_plot(output_basename)
    x = np.arange(random.randint(5, 20))
    y = np.cumsum(np.random.randn(len(x))) + 50
    ax.plot(x, y, marker='o')
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    series_data = [(str(xi), float(yi)) for xi, yi in zip(x, y)]
    schema = AnchorChartSchema(chart_type="line", axes={"x": {"label": "Time"}, "y": {"label": "Value"}}, 
                               series=[AnchorSeries(label="Trend", data_points=series_data)])
    save_and_close(fig, output_basename, schema)

def generate_scatter_chart(output_basename=None):
    if output_basename is None: output_basename = f"chart_{uuid.uuid4().hex[:8]}_scatter"
    fig, ax = init_plot(output_basename)
    x = np.random.uniform(0, 100, 50)
    y = 2*x + np.random.normal(0, 10, 50)
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    series_data = [(str(round(xi,2)), float(yi)) for xi, yi in zip(x, y)]
    schema = AnchorChartSchema(chart_type="scatter", axes={"x": {"label": "X Axis"}, "y": {"label": "Y Axis"}}, 
                               series=[AnchorSeries(label="Points", data_points=series_data)])
    save_and_close(fig, output_basename, schema)
