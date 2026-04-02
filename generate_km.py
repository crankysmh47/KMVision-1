import matplotlib
matplotlib.use('Agg') # CRITICAL for Thread/Process Safety
import matplotlib.pyplot as plt
import os
import uuid
import json
import random
import numpy as np
import pandas as pd
import gc
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from schemas import KMChartSchema, KMAxes, Axis, KMArm

# Optimized Configs
OUTPUT_DIR = "dataset"
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

DEFAULT_FONT = 'DejaVu Sans'
COLORS = [
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],  # Tableau 10 base
    ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'],  # Set1
    ['#000000', '#555555', '#888888', '#bbbbbb'],  # Grayscale
    ['#003f5c', '#7a5195', '#ef5675', '#ffa600']   # Distinct mix
]
LINESTYLES = ['-', '--', '-.', ':']

def generate_arm_data(n_samples, scale, shape, censor_rate):
    """
    Generates synthetic survival data based on Weibull distribution.
    Parameters scale and shape control the Weibull curve.
    censor_rate adds a uniform censoring probability framework.
    """
    actual_lifetimes = scale * np.random.weibull(shape, n_samples)
    censor_times = np.random.uniform(0, scale * (1 + censor_rate*2), n_samples)
    
    observed_times = np.minimum(actual_lifetimes, censor_times)
    event_observed = (actual_lifetimes <= censor_times).astype(int)
    
    return observed_times, event_observed

def generate_km_chart(output_basename=None):
    if output_basename is None:
        output_basename = f"chart_{uuid.uuid4().hex[:8]}_km"
    
    # RE-SEED ALL RNGS for process safety
    seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(seed)
    random.seed(seed)

    # Use single reliable font to avoid font search overhead
    plt.rcParams.update({'font.family': DEFAULT_FONT})
    
    fig_width = random.uniform(6, 10)
    fig_height = random.uniform(5, 8)
    dpi = random.choice([100, 150, 200, 300]) 
    
    # Ensure maximum dimension is capped around 1024 to prevent huge memory spikes
    if fig_width * dpi > 1024:
        dpi = int(1024 / fig_width)
    if fig_height * dpi > 1024:
        dpi = int(1024 / fig_height)
        
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    n_arms = random.randint(2, 4)
    palette = random.choice(COLORS)
    include_grid = random.choice([True, False])
    show_risk_table = random.random() < 0.5
    
    ax.set_ylabel("Survival Probability", fontsize=random.randint(10, 14))
    ax.set_xlabel("Time (Months)", fontsize=random.randint(10, 14))
    
    if include_grid:
        ax.grid(True, linestyle=random.choice(['-', '--', ':']), alpha=random.uniform(0.3, 0.7))
        
    arms_schema = []
    kmfs = []
    
    for i in range(n_arms):
        n_samples = random.randint(50, 200)
        scale = random.uniform(20, 100)
        shape = random.uniform(0.8, 1.5)
        censor_rate = random.uniform(0.1, 0.5)
        
        T, E = generate_arm_data(n_samples, scale, shape, censor_rate)
        
        kmf = KaplanMeierFitter()
        treatment_label = f"Treatment {chr(65+i)}"
        kmf.fit(T, event_observed=E, label=treatment_label)
        kmfs.append(kmf)
        
        color = palette[i % len(palette)]
        linestyle = random.choice(LINESTYLES)
        
        kmf.plot_survival_function(
            ax=ax, 
            ci_show=False, 
            show_censors=True, 
            censor_styles={'marker': '|', 'ms': random.randint(6, 12), 'mew': random.randint(1, 2)},
            color=color,
            linestyle=linestyle,
            label=treatment_label
        )
        
        # Extract precise coordinates for JSON
        survival_df = kmf.survival_function_
        coords = []
        for time_idx, row in survival_df.iterrows():
            prob = float(row.iloc[0])
            coords.append((float(time_idx), prob))
            
        # Extract precise censoring ticks
        event_table = kmf.event_table
        censored_df = event_table[event_table['censored'] > 0]
        censoring_ticks = [float(idx) for idx in censored_df.index]
        
        # Build Arm Schema
        arms_schema.append(KMArm(
            treatment_label=treatment_label,
            coordinates=coords,
            censoring_ticks=censoring_ticks
        ))
        
    ax.legend(loc=random.choice(['best', 'upper right', 'lower left', 'lower right']))
    
    # Tick mark density block
    x_max = ax.get_xlim()[1]
    step_sizes = [5, 10, 20, 25, 50]
    valid_steps = [s for s in step_sizes if (x_max / s) <= 15 and s > 0]
    if valid_steps:
        step = random.choice(valid_steps)
        ax.set_xticks(np.arange(0, x_max + step, step))
        
    ax.set_ylim([0.0, 1.05])
    
    # Optional At Risk Table Below X Axis (50% probability)
    if show_risk_table:
        add_at_risk_counts(*kmfs, ax=ax)
    
    # Capture complete schema object
    schema = KMChartSchema(
        axes=KMAxes(
            x=Axis(label="Time (Months)", max_value=float(x_max)),
            y=Axis(label="Survival Probability", max_value=1.0)
        ),
        arms=arms_schema
    )
    
    # Save Image to Disk
    img_path = os.path.join(OUTPUT_DIR, "images", f"{output_basename}.png")
    plt.savefig(img_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig) # Clear specific figure
    plt.close('all') # Catch-all
    gc.collect() # Force garbage collection
    
    # Save JSON Ground Truth to Disk
    json_path = os.path.join(OUTPUT_DIR, "labels", f"{output_basename}.json")
    try:
        json_output = schema.model_dump_json(indent=2)
    except AttributeError:
        # Fallback to older pydantic versions
        json_output = schema.json(indent=2)
        
    with open(json_path, 'w') as f:
        f.write(json_output)

if __name__ == "__main__":
    # Test stub to demonstrate it works standalone
    print("Testing single KM chart generation...")
    generate_km_chart("test_chart_km")
    print(f"Chart generated to dataset/images/test_chart_km.png")
