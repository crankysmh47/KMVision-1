import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, uuid, random, json
import numpy as np
import gc
from schemas import ForestChartSchema, ForestStudy, WaterfallChartSchema, WaterfallBar
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
    return fig, ax, seed

def generate_forest_plot(output_basename=None):
    if output_basename is None: output_basename = f"chart_{uuid.uuid4().hex[:8]}_forest"
    fig, ax, _ = init_plot(output_basename)
    
    n_studies = random.randint(3, 10)
    studies = []
    
    y_pos = np.arange(n_studies, 0, -1)
    study_prefix = random.choice(["Study", "Trial", "Clinic", "Center", "Site", "Hospital"])
    
    for i in range(n_studies):
        study_label = f"{study_prefix} {chr(65+i)}" if random.random() > 0.3 else f"{study_prefix} {random.randint(1, 100)}"
        ratio = max(0.1, random.normalvariate(1.0, 0.4))
        error_margin = random.uniform(0.1, 0.5)
        
        ci_lower = max(0.01, ratio - error_margin)
        ci_upper = ratio + error_margin
        
        studies.append(ForestStudy(study_label=study_label, ratio_value=ratio, ci_lower=ci_lower, ci_upper=ci_upper))
        
        ax.errorbar(ratio, y_pos[i], xerr=[[ratio - ci_lower], [ci_upper - ratio]], 
                    fmt='s', color='black', capsize=random.randint(3, 8), markersize=random.randint(5, 10))
        
    overall_ratio = np.mean([s.ratio_value for s in studies])
    overall_err = np.mean([s.ci_upper - s.ratio_value for s in studies]) / np.sqrt(n_studies)
    overall_ci_lower = overall_ratio - overall_err
    overall_ci_upper = overall_ratio + overall_err
    
    overall_effect = ForestStudy(study_label="Overall", ratio_value=overall_ratio, ci_lower=overall_ci_lower, ci_upper=overall_ci_upper)
    
    ax.errorbar(overall_ratio, 0, xerr=[[overall_ratio - overall_ci_lower], [overall_ci_upper - overall_ratio]], 
                fmt='D', color='red', capsize=random.randint(3,8), markersize=random.randint(6, 12))
    
    ax.axvline(1.0, linestyle=random.choice(['--', '-.', ':']), color='gray')
    ax.set_yticks(np.append(y_pos, 0))
    ax.set_yticklabels([s.study_label for s in studies] + ["Overall Effect"])
    
    x_label = generate_label()
    ax.set_xlabel(x_label)
    if random.choice([True, False]): ax.grid(True, axis='x', alpha=0.3)
        
    forest_img_dir = os.path.join(OUTPUT_DIR, "images", "forest")
    os.makedirs(forest_img_dir, exist_ok=True)
    img_path = os.path.join(forest_img_dir, f"{output_basename}.png")
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    plt.close('all')
    gc.collect()
    
    schema = ForestChartSchema(axes={"x": {"label": x_label}}, studies=studies, overall_effect=overall_effect)
    forest_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "forest")
    os.makedirs(forest_lbl_dir, exist_ok=True)
    with open(os.path.join(forest_lbl_dir, f"{output_basename}.json"), 'w') as f:
        try: f.write(schema.model_dump_json(indent=2))
        except: f.write(schema.json(indent=2))

def generate_waterfall_plot(output_basename=None):
    if output_basename is None: output_basename = f"chart_{uuid.uuid4().hex[:8]}_waterfall"
    fig, ax, _ = init_plot(output_basename)
    
    n_patients = random.randint(20, 50)
    values = np.random.normal(-10, 30, n_patients)
    values = np.clip(values, -100, 100)
    values.sort()
    values = values[::-1] # descending
    
    bars = []
    x_pos = np.arange(n_patients)
    
    colors = []
    for val in values:
        if val > 20: colors.append(COLORS[3]) # red 
        elif val < -30: colors.append(COLORS[2]) # green 
        else: colors.append(COLORS[0]) # blue 
            
    ax.bar(x_pos, values, color=colors, width=random.uniform(0.6, 1.0))
    
    for i, val in enumerate(values):
        bars.append(WaterfallBar(label=f"Patient {i+1}", value=float(val)))
        
    ax.axhline(0, color='black', linewidth=1)
    if random.choice([True, False]): ax.axhline(-30, color='red', linestyle='--', alpha=0.6)
    
    ax.set_xticks([])
    x_label = generate_label()
    ax.set_xlabel(x_label)
    y_label = generate_label()
    ax.set_ylabel(y_label)
    
    waterfall_img_dir = os.path.join(OUTPUT_DIR, "images", "waterfall")
    os.makedirs(waterfall_img_dir, exist_ok=True)
    img_path = os.path.join(waterfall_img_dir, f"{output_basename}.png")
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight')
    plt.close('all')
    
    schema = WaterfallChartSchema(axes={"y": {"label": y_label}}, bars=bars)
    waterfall_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "waterfall")
    os.makedirs(waterfall_lbl_dir, exist_ok=True)
    with open(os.path.join(waterfall_lbl_dir, f"{output_basename}.json"), 'w') as f:
        try: f.write(schema.model_dump_json(indent=2))
        except: f.write(schema.json(indent=2))
