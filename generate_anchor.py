import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, uuid, random
import numpy as np
import gc

from schemas import AnchorChartSchema, AnchorSeries, AnchorDataPoint
from lexical_engine import generate_label

OUTPUT_DIR = r"C:\sem4\KMVision-1 Data\dataset"
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

DEFAULT_FONT = 'DejaVu Sans'
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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
    plt.savefig(img_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    plt.close('all')
    gc.collect()
    with open(os.path.join(OUTPUT_DIR, "labels", f"{output_basename}.json"), 'w') as f:
        try: f.write(schema.model_dump_json(indent=2))
        except: f.write(schema.json(indent=2))

def build_data_points(x_vals, y_vals):
    return [AnchorDataPoint(x=x, y=float(y)) for x, y in zip(x_vals, y_vals)]

def generate_random_anchor(output_basename=None):
    if output_basename is None: 
        output_basename = f"chart_{uuid.uuid4().hex[:8]}_anchor"
        
    chart_choices = ['simple_bar', 'stacked_bar', 'multi_line', 'dual_axis_combo', 'scatter']
    probs = [0.20, 0.20, 0.20, 0.25, 0.15]
    
    chart_type = random.choices(chart_choices, weights=probs, k=1)[0]
    
    fig, ax = init_plot(output_basename)
    
    series_list = []
    axes_schema = {}
    
    if chart_type == 'simple_bar':
        n_bars = random.randint(3, 10)
        x_cats = [generate_label() for _ in range(n_bars)]
        y_vals = np.abs(np.random.normal(50, 20, n_bars))
        
        ax.bar(x_cats, y_vals, color=random.choice(COLORS))
        
        x_label = generate_label()
        y_label = generate_label()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        axes_schema = {"x": {"label": x_label}, "y": {"label": y_label}}
        series_list.append(AnchorSeries(series_name="Data", series_type="bar", data=build_data_points(x_cats, y_vals)))
        
    elif chart_type == 'stacked_bar':
        n_bars = random.randint(3, 8)
        n_stacks = random.randint(2, 4)
        x_cats = [generate_label() for _ in range(n_bars)]
        bottom = np.zeros(n_bars)
        
        for i in range(n_stacks):
            y_vals = np.abs(np.random.uniform(10, 50, n_bars))
            s_name = generate_label()
            ax.bar(x_cats, y_vals, bottom=bottom, label=s_name, color=COLORS[i%len(COLORS)])
            bottom += y_vals
            series_list.append(AnchorSeries(series_name=s_name, series_type="bar", data=build_data_points(x_cats, y_vals)))
            
        ax.legend()
        x_label = generate_label()
        y_label = generate_label()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        axes_schema = {"x": {"label": x_label}, "y": {"label": y_label}}
        
    elif chart_type == 'multi_line':
        n_lines = random.randint(2, 5)
        n_pts = random.randint(5, 50)
        x_vals = np.arange(n_pts)
        
        for i in range(n_lines):
            offset = random.uniform(0, 100)
            scale = random.uniform(1, 10)
            y_vals = np.cumsum(np.random.randn(n_pts) * scale) + offset
            s_name = generate_label()
            ax.plot(x_vals, y_vals, label=s_name, marker=random.choice(['o', 's', '^', None]))
            series_list.append(AnchorSeries(series_name=s_name, series_type="line", data=build_data_points(x_vals, y_vals)))
            
        ax.legend()
        x_label = generate_label()
        y_label = generate_label()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        axes_schema = {"x": {"label": x_label}, "y": {"label": y_label}}
        
    elif chart_type == 'dual_axis_combo':
        n_pts = random.randint(5, 12)
        x_cats = [generate_label() for _ in range(n_pts)]
        
        ax2 = ax.twinx()
        
        y1_vals = np.abs(np.random.uniform(10, 100, n_pts))
        s1_name = generate_label()
        ax.bar(x_cats, y1_vals, color='lightblue', alpha=0.7, label=s1_name)
        
        y2_vals = np.cumsum(np.random.randn(n_pts) * 5) + 50
        s2_name = generate_label()
        ax2.plot(x_cats, y2_vals, color='red', marker='D', linewidth=2, label=s2_name)
        
        x_label = generate_label()
        y1_label = generate_label()
        y2_label = generate_label()
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y1_label)
        ax2.set_ylabel(y2_label)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
        
        ax.set_xticks(range(len(x_cats)))
        ax.set_xticklabels(x_cats, rotation=45, ha='right', fontsize=8)
        
        series_list.append(AnchorSeries(series_name=s1_name, series_type="bar", data=build_data_points(x_cats, y1_vals)))
        series_list.append(AnchorSeries(series_name=s2_name, series_type="line", data=build_data_points(x_cats, y2_vals)))
        axes_schema = {"x": {"label": x_label}, "y1": {"label": y1_label}, "y2": {"label": y2_label}}
        
    elif chart_type == 'scatter':
        n_clusters = random.randint(1, 3)
        for i in range(n_clusters):
            n_pts = random.randint(20, 100)
            x_vals = np.random.normal(random.uniform(0, 100), random.uniform(5, 20), n_pts)
            y_vals = np.random.normal(random.uniform(0, 100), random.uniform(5, 20), n_pts)
            s_name = generate_label()
            
            if random.random() > 0.5:
                y_vals += x_vals * random.uniform(-1, 1)
                
            ax.scatter(x_vals, y_vals, alpha=0.6, label=s_name, edgecolors='w')
            series_list.append(AnchorSeries(series_name=s_name, series_type="scatter", data=build_data_points(x_vals, y_vals)))
            
        if n_clusters > 1: ax.legend()
        x_label = generate_label()
        y_label = generate_label()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        axes_schema = {"x": {"label": x_label}, "y": {"label": y_label}}

    schema = AnchorChartSchema(chart_type=chart_type, axes=axes_schema, series=series_list)
    save_and_close(fig, output_basename, schema)

def generate_bar_chart(output_basename=None): generate_random_anchor(output_basename)
def generate_line_chart(output_basename=None): generate_random_anchor(output_basename)
def generate_scatter_chart(output_basename=None): generate_random_anchor(output_basename)
