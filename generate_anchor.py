import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, uuid, random
import numpy as np
import pandas as pd
import seaborn as sns
import traceback
import gc

from schemas import AnchorChartSchema, AnchorSeries
from lexical_engine import generate_label

OUTPUT_DIR = r"C:\sem4\KMVision-1 Data\dataset"
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

DEFAULT_FONT = 'DejaVu Sans'

CHARTQA_DIR = r"C:\sem4\KMVision-1 Data\ChartQA\ChartQA Dataset\train\tables"
if os.path.exists(CHARTQA_DIR):
    CHARTQA_TABLES = [os.path.join(CHARTQA_DIR, f) for f in os.listdir(CHARTQA_DIR) if f.endswith('.csv')]
else:
    CHARTQA_TABLES = []

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
    # Using bbox_inches='tight' below already handles layout, 
    # tight_layout() here throws UserWarnings on extreme ChartQA string sizes.
    plt.savefig(img_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    plt.close('all')
    gc.collect()
    with open(os.path.join(OUTPUT_DIR, "labels", f"{output_basename}.json"), 'w') as f:
        try: f.write(schema.model_dump_json(indent=2))
        except: f.write(schema.json(indent=2))

def fetch_valid_chartqa_data():
    """
    Randomly searches for a valid CSV containing at least 1 string column and 1 num column.
    """
    for _ in range(10): # Try 10 times to find a valid CSV
        if not CHARTQA_TABLES:
            break
        csv_path = random.choice(CHARTQA_TABLES)
        try:
            df = pd.read_csv(csv_path)
            # drop fully empty rows/cols
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)
            
            # Simple heuristic
            cat_cols = [c for c in df.columns if df[c].dtype == 'object']
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            
            if len(cat_cols) > 0 and len(num_cols) > 0:
                # Force specific limits so it plots reasonably
                df = df.head(15).copy() # Cap to 15 items for visual sanity
                # Select the first categorical and first numerical
                x_col = cat_cols[0]
                y_col = num_cols[0]
                
                # Fill NAs and aggressively clean raw ChartQA string formats (removing tabs/newlines avoiding Missing Glyph warnings)
                df[x_col] = df[x_col].fillna("Unknown").astype(str).str.replace(r'[\t\n\r]', ' ', regex=True).str.slice(0, 40)
                df[y_col] = df[y_col].fillna(0.0).astype(float)
                
                if len(df) > 2:
                    return df[x_col].tolist(), df[y_col].tolist()
        except:
            continue
    raise ValueError("Failed to fetch viable ChartQA CSV")

def generate_bar_chart(output_basename=None):
    if output_basename is None: output_basename = f"chart_{uuid.uuid4().hex[:8]}_bar"
    try:
        categories, values = fetch_valid_chartqa_data()
        fig, ax = init_plot(output_basename)
        sns.barplot(x=categories, y=values, ax=ax, hue=categories, palette="viridis", legend=False)
        
        y_label = generate_label()
        x_label = generate_label()
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        # Avoid label overlap by rotating
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        series_data = [(str(cat), float(val)) for cat, val in zip(categories, values)]
        schema = AnchorChartSchema(chart_type="bar", axes={"x": {"label": x_label}, "y": {"label": y_label}}, 
                                   series=[AnchorSeries(label="Data", data_points=series_data)])
        save_and_close(fig, output_basename, schema)
    except Exception as e:
        pass # silent try except

def generate_line_chart(output_basename=None):
    if output_basename is None: output_basename = f"chart_{uuid.uuid4().hex[:8]}_line"
    try:
        x, y = fetch_valid_chartqa_data()
        fig, ax = init_plot(output_basename)
        
        ax.plot(x, y, marker='o')
        
        x_label = generate_label()
        y_label = generate_label()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        series_data = [(str(xi), float(yi)) for xi, yi in zip(x, y)]
        schema = AnchorChartSchema(chart_type="line", axes={"x": {"label": x_label}, "y": {"label": y_label}}, 
                                   series=[AnchorSeries(label="Trend", data_points=series_data)])
        save_and_close(fig, output_basename, schema)
    except Exception as e:
        pass # silent try except

def generate_scatter_chart(output_basename=None):
    if output_basename is None: output_basename = f"chart_{uuid.uuid4().hex[:8]}_scatter"
    try:
        x, y = fetch_valid_chartqa_data()
        fig, ax = init_plot(output_basename)
        # Usually scatter needs 2 numeric blocks. ChartQA validation guarantees 1 num, 1 cat.
        # So we scatter cat vs num.
        ax.scatter(x, y, alpha=0.7)
        
        x_label = generate_label()
        y_label = generate_label()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        series_data = [(str(xi), float(yi)) for xi, yi in zip(x, y)]
        schema = AnchorChartSchema(chart_type="scatter", axes={"x": {"label": x_label}, "y": {"label": y_label}}, 
                                   series=[AnchorSeries(label="Data Points", data_points=series_data)])
        save_and_close(fig, output_basename, schema)
    except Exception as e:
        pass # silent try except

