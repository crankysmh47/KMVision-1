import matplotlib
matplotlib.use('Agg') # CRITICAL for Thread/Process Safety
import matplotlib.pyplot as plt
import os
import uuid
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import gc
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from schemas import KMChartSchema, KMAxes, Axis, KMArm, AtRiskTimepoint
from lexical_engine import generate_label

# Optimized Configs
OUTPUT_DIR = r"C:\sem4\KMVision-1 Data\dataset"
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

TIME_UNITS = {
    'months': {'weight': 0.60, 'factor': 1.0, 'ticks': [5, 10, 20, 25, 50]},
    'weeks':  {'weight': 0.25, 'factor': 4.345, 'ticks': [10, 20, 50]},
    'days':   {'weight': 0.15, 'factor': 30.44, 'ticks': [50, 100, 200]},
}
TITLE_TEMPLATES = [
    "{y}: {a0} versus {a1}",
    "Kaplan-Meier curve of {y}",
    "{y} stratified by treatment group",
    "Effect of {a1} on {y}",
    "Comparison of {a0} and {a1}: {y}",
    "Survival analysis of {y}",
]

def pick_time_unit():
    units = list(TIME_UNITS.keys())
    weights = [TIME_UNITS[u]['weight'] for u in units]
    return random.choices(units, weights=weights, k=1)[0]

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

def compute_effect_stats(arm_data):
    T_all = np.concatenate([t for t, _ in arm_data])
    E_all = np.concatenate([e for _, e in arm_data])
    groups = np.concatenate([np.full(len(t), i) for i, (t, _) in enumerate(arm_data)])

    p_value = float(multivariate_logrank_test(T_all, groups, E_all).p_value)

    df_cox = pd.DataFrame({'duration': T_all, 'event': E_all})
    for j in range(1, len(arm_data)):
        df_cox[f'arm_{j}'] = (groups == j).astype(int)
    try:
        cph = CoxPHFitter()
        cph.fit(df_cox, 'duration', 'event')
        row = cph.summary.loc['arm_1']
        hr = float(np.exp(float(row['coef'])))
        ci_lower = float(row['exp(coef) lower 95%'])
        ci_upper = float(row['exp(coef) upper 95%'])
    except Exception:
        hr, ci_lower, ci_upper = 1.0, 1.0, 1.0

    return round(hr, 3), round(ci_lower, 3), round(ci_upper, 3), p_value

def build_at_risk_table(risk_times, arm_labels, arm_data):
    table = []
    for t in risk_times:
        # End-of-period convention, matching lifelines' add_at_risk_counts
        # rendering (at_risk - removed at t): subjects still observed after t.
        counts = {label: int(np.sum(T > t)) for label, (T, _) in zip(arm_labels, arm_data)}
        table.append(AtRiskTimepoint(timepoint=float(t), counts=counts))
    return table

def generate_km_chart(output_basename=None, output_dir=None):
    if output_basename is None:
        output_basename = f"chart_{uuid.uuid4().hex[:8]}_km"
    
    # RE-SEED ALL RNGS for process safety
    seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(seed)
    random.seed(seed)

    time_unit = pick_time_unit()
    unit_factor = TIME_UNITS[time_unit]['factor']

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
    
    y_label = generate_label()
    x_label = f"{random.choice(['Time', 'Follow-up time', 'Survival time'])} ({time_unit})"
    
    ax.set_ylabel(y_label, fontsize=random.randint(10, 14))
    ax.set_xlabel(x_label, fontsize=random.randint(10, 14))
    
    if include_grid:
        ax.grid(True, linestyle=random.choice(['-', '--', ':']), alpha=random.uniform(0.3, 0.7))
        
    arms_schema = []
    kmfs = []
    arm_data = []
    
    style_choice = random.choice(['Treatment', 'Drug', 'Cohort', 'Group', 'Dose'])
    if style_choice == 'Treatment':
        arm_labels = [f"Treatment {chr(65+i)}" for i in range(4)]
    elif style_choice == 'Drug':
        arm_labels = [f"Drug {chr(88+i)}" for i in range(4)]
    elif style_choice == 'Cohort':
        arm_labels = [f"Cohort {i+1}" for i in range(4)]
    elif style_choice == 'Group':
        arm_labels = [f"Group {i+1}" for i in range(4)]
    else:
        arm_labels = ["Placebo", "Low Dose", "Medium Dose", "High Dose"]
        
    for i in range(n_arms):
        n_samples = random.randint(50, 200)
        scale = random.uniform(20, 100) * unit_factor
        shape = random.uniform(0.8, 1.5)
        censor_rate = random.uniform(0.1, 0.5)
        
        T, E = generate_arm_data(n_samples, scale, shape, censor_rate)
        arm_data.append((T, E))
        
        kmf = KaplanMeierFitter()
        treatment_label = arm_labels[i]
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
    valid_steps = [s for s in TIME_UNITS[time_unit]['ticks'] if (x_max / s) <= 15 and s > 0]
    if not valid_steps:
        valid_steps = [max(1, int(x_max / 10))]
    step = random.choice(valid_steps)
    ax.set_xticks(np.arange(0, x_max + step, step))
        
    ax.set_ylim([0.0, 1.05])
    
    # Optional At Risk Table Below X Axis (50% probability)
    at_risk_table = []
    if show_risk_table:
        tick_vals = [float(t) for t in ax.get_xticks() if 0 < float(t) < x_max]
        if len(tick_vals) > 8:
            idx = np.round(np.linspace(0, len(tick_vals) - 1, 8)).astype(int)
            tick_vals = [tick_vals[k] for k in dict.fromkeys(idx)]
        add_at_risk_counts(*kmfs, ax=ax, xticks=tick_vals)
        at_risk_table = build_at_risk_table(tick_vals, arm_labels[:n_arms], arm_data)
    
    hazard_ratio, ci_lower, ci_upper, p_value = compute_effect_stats(arm_data)
    
    title = random.choice(TITLE_TEMPLATES).format(
        y=y_label, a0=arm_labels[0], a1=arm_labels[1]
    )
    ax.set_title(title, fontsize=random.randint(11, 15))
    
    # Capture complete schema object
    schema = KMChartSchema(
        title=title,
        time_unit=time_unit,
        hazard_ratio=hazard_ratio,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        axes=KMAxes(
            x=Axis(label=x_label, max_value=float(x_max)),
            y=Axis(label=y_label, max_value=1.0)
        ),
        arms=arms_schema,
        at_risk_table=at_risk_table
    )
    
    out_root = Path(output_dir) if output_dir else Path(OUTPUT_DIR)
    
    # Save Image to Disk
    km_img_dir = os.path.join(out_root, "images", "km")
    os.makedirs(km_img_dir, exist_ok=True)
    img_path = os.path.join(km_img_dir, f"{output_basename}.png")
    plt.savefig(img_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig) # Clear specific figure
    plt.close('all') # Catch-all
    gc.collect() # Force garbage collection
    
    # Save JSON Ground Truth to Disk
    km_lbl_dir = os.path.join(out_root, "labels", "km")
    os.makedirs(km_lbl_dir, exist_ok=True)
    json_path = os.path.join(km_lbl_dir, f"{output_basename}.json")
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
