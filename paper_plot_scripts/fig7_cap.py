import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# ==================================================================================
# CONFIGURATION
# ==================================================================================

BASE_BENCH_DIR = "benchmark_output/fig7"
DATASETS = ["llama3_8b_sharegpt", "llama3_8b_azconv", "llama3_8b_azcode"]
VIOLATION_THRESHOLD = 1.0

SLO_THRESHOLDS = {0: 6.0, 1: 600.0, 2: 1800.0}

LABEL_MAPPING = {'fcfs': 'Sarathi-FCFS', 'edf': 'Sarathi-EDF', 'deadline': 'Niyama'}
COLOR_MAPPING = {'fcfs': '#A0522D', 'edf': '#4682B4', 'deadline': '#6B8E23'}

# ==================================================================================
# CORE LOGIC
# ==================================================================================

def calculate_max_tier_violation(run_dir):
    csv_path = os.path.join(run_dir, "replica_0", "sequence_metrics.csv")
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path)
        if 'request_tier' not in df.columns: df['request_tier'] = 2
        else: df['request_tier'] = df['request_tier'].replace(-1, 2)

        tier_violations = []
        for tier in [0, 1, 2]:
            tier_df = df[df['request_tier'] == tier]
            if len(tier_df) == 0: continue
            metric_col = 'prefill_e2e_time' if tier == 0 else 'request_e2e_time'
            slo_val = SLO_THRESHOLDS[tier]
            violation_count = len(tier_df[tier_df[metric_col] > slo_val])
            tier_violations.append((violation_count / len(tier_df)) * 100.0)
        return max(tier_violations) if tier_violations else 0.0
    except: return None

def interpolate_capacity(q_low, v_low, q_high, v_high):
    """Linear interpolation to find QPS at exactly 1.0% violation."""
    if v_high == v_low: return q_low
    val = q_low + (1.0 - v_low) * (q_high - q_low) / (v_high - v_low)
    return round(val, 2)

# ==================================================================================
# PLOTTING
# ==================================================================================

def plot_single_workload(dataset_name, capacities):
    filename = f"capacity_{dataset_name}.pdf"
    schedulers = ['fcfs', 'edf', 'deadline']
    values = [capacities.get(s, 0.0) for s in schedulers]
    labels = [LABEL_MAPPING.get(s, s) for s in schedulers]
    colors = [COLOR_MAPPING.get(s, 'gray') for s in schedulers]
    
    plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(7, 5))
    indices = np.arange(len(schedulers))
    bars = ax.bar(indices, values, 0.6, color=colors, edgecolor='black', zorder=3)
    
    clean_title = dataset_name.replace("llama3_8b_", "").capitalize()
    ax.set_ylabel('Goodput (QPS)', fontweight='bold')
    ax.set_title(f'Max Goodput ({clean_title})', fontweight='bold', pad=15)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    # Add Value Labels with 2 decimal places
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(values)*0.01),
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ==================================================================================
# MAIN
# ==================================================================================

def main():
    for dataset in DATASETS:
        dataset_path = os.path.join(BASE_BENCH_DIR, dataset)
        if not os.path.exists(dataset_path): continue

        print(f"\n{'='*70}\nWORKLOAD: {dataset}\n{'='*70}")
        raw_results = {}

        for d in os.listdir(dataset_path):
            if '_' in d:
                parts = d.split('_')
                try:
                    qps = float(parts[-1])
                    sched = "_".join(parts[:-1])
                    base_path = os.path.join(dataset_path, d)
                    subdirs = [os.path.join(base_path, sd) for sd in os.listdir(base_path)]
                    if not subdirs: continue
                    run_dir = max(subdirs, key=os.path.getmtime)
                    viol = calculate_max_tier_violation(run_dir)
                    if viol is not None:
                        if sched not in raw_results: raw_results[sched] = []
                        raw_results[sched].append((qps, viol))
                except: continue

        final_plot_caps = {}
        for sched in ['fcfs', 'edf', 'deadline']:
            if sched not in raw_results: continue
            data = sorted(raw_results[sched], key=lambda x: x[0])
            
            print(f"\n[{LABEL_MAPPING[sched].upper()}]")
            
            low_qps, low_viol = 0.0, 0.0
            found = False
            
            for q, v in data:
                if v <= VIOLATION_THRESHOLD:
                    low_qps, low_viol = q, v
                else:
                    interpolated = interpolate_capacity(low_qps, low_viol, q, v)
                    # Print range and result to 2 decimals
                    print(f"  RANGE: [{low_qps:.2f} (pass) - {q:.2f} (fail)]")
                    print(f"  RESULT: 1% estimated at {interpolated:.2f} QPS")
                    final_plot_caps[sched] = interpolated
                    found = True
                    break
            
            if not found and data:
                max_q = data[-1][0]
                print(f"  RANGE: All Passed. Max tested: {max_q:.2f}")
                final_plot_caps[sched] = round(max_q, 2)

        if final_plot_caps:
            plot_single_workload(dataset, final_plot_caps)

if __name__ == "__main__":
    main()