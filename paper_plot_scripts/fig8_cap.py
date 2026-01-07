import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys

# ==================================================================================
# CONFIGURATION
# ==================================================================================

# Root directory for Figure 8 results (Updated to 'azconv')
BASE_DIR = "benchmark_output/fig8/llama3_8b_azconv"

# Violation Threshold (Strict 1%)
VIOLATION_THRESHOLD = 1.0

# SLO Configurations
SLO_THRESHOLDS = {
    0: 6.0,    # Tier 0: 6s (Prefill E2E)
    1: 600.0,  # Tier 1: 600s (Request E2E)
    2: 1800.0  # Tier 2: 1800s (Request E2E)
}

# Plotting Configuration
OUTPUT_PLOT_FILENAME = "fig8_capacity.pdf"

# Mapping internal names to Paper labels (Fig 8 Style)
LABEL_MAPPING = {
    'fcfs': 'Disagg-FCFS',
    'edf': 'Disagg-EDF',
    'deadline_no_dynamic_chunking': 'Disagg-Niyama'
}

# Colors matching Figure 8 (Approximate from screenshot)
COLOR_MAPPING = {
    'fcfs': '#A0522D',                       # Sienna (Brown)
    'edf': '#4F6D7A',                        # Slate/Blue-Grey
    'deadline_no_dynamic_chunking': '#6B8E23' # OliveDrab
}

# ==================================================================================
# PARSING LOGIC
# ==================================================================================

def get_latest_run_dir(sched_type, qps):
    """Finds the most recent timestamped run for a given scheduler and QPS."""
    base_path = os.path.join(BASE_DIR, f"{sched_type}_{qps}")
    if not os.path.exists(base_path):
        return None
    
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) 
               if os.path.isdir(os.path.join(base_path, d))]
    
    if not subdirs:
        return None
        
    return max(subdirs, key=os.path.getmtime)

def calculate_violation(sched_type, qps, run_dir):
    """Parses sequence_metrics.csv and calculates Overall Violation %."""
    csv_path = os.path.join(run_dir, "replica_0", "sequence_metrics.csv")
    
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    
    # Map missing tiers to Tier 2
    if 'request_tier' not in df.columns:
        df['request_tier'] = 2
    else:
        df['request_tier'] = df['request_tier'].replace(-1, 2)

    tier_violations = []

    for tier in [0, 1, 2]:
        tier_df = df[df['request_tier'] == tier]
        count = len(tier_df)
        
        if count == 0:
            tier_violations.append(0.0)
            continue
            
        metric_col = 'prefill_e2e_time' if tier == 0 else 'request_e2e_time'
        slo_val = SLO_THRESHOLDS[tier]
        
        violation_count = len(tier_df[tier_df[metric_col] > slo_val])
        pct = (violation_count / count) * 100.0
        tier_violations.append(pct)

    # Overall Violation is the average of the 3 tiers
    overall_viol = sum(tier_violations) / 3.0
    return overall_viol

# ==================================================================================
# PLOTTING LOGIC
# ==================================================================================

def plot_capacity_bars(capacities):
    """
    Generates the grouped bar chart for Figure 8.
    """
    print(f"\n>>> Generating Plot: {OUTPUT_PLOT_FILENAME}...")
    
    # Order: FCFS, EDF, Niyama
    schedulers = ['fcfs', 'edf', 'deadline_no_dynamic_chunking']
    
    # Extract values
    values = [capacities.get(s, 0.0) for s in schedulers]
    labels = [LABEL_MAPPING.get(s, s) for s in schedulers]
    colors = [COLOR_MAPPING.get(s, 'gray') for s in schedulers]
    
    # Setup Plot (Dark background style not applied to save ink, keeping standard white)
    plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Position: We only have 1 group (Llama3-8B), centered at x=0
    x_pos = np.arange(1)  # [0]
    width = 0.25          # Width of individual bars
    
    # Plot bars
    # Bar 1: FCFS (Offset left)
    rects1 = ax.bar(x_pos - width, [values[0]], width, label=labels[0], 
                    color=colors[0], edgecolor='white', hatch='--')
    
    # Bar 2: EDF (Center)
    rects2 = ax.bar(x_pos, [values[1]], width, label=labels[1], 
                    color=colors[1], edgecolor='white', hatch='//')
    
    # Bar 3: Niyama (Offset right)
    rects3 = ax.bar(x_pos + width, [values[2]], width, label=labels[2], 
                    color=colors[2], edgecolor='white')
    
    # Formatting
    ax.set_ylabel('Goodput (QPS)', fontweight='bold', fontsize=14)
    ax.set_ylim(0, max(values) * 1.3) # Add headroom for labels
    
    # X-Axis Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Llama3-8B\n(TP1-A100)'], fontweight='bold', fontsize=12)
    
    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # Value Labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Legend
    ax.legend(loc='upper left', frameon=True, fontsize=11, ncol=1)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILENAME)
    print(f">>> Plot Saved to {OUTPUT_PLOT_FILENAME}")

# ==================================================================================
# MAIN ANALYSIS
# ==================================================================================

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Error: Directory {BASE_DIR} not found. Check your BASE_DIR setting.")
        sys.exit(1)

    print(f"{'SCHEDULER':<35} | {'QPS':<6} | {'VIOLATION %':<12} | {'STATUS'}")
    print("-" * 75)

    results = {}

    # 1. SCAN DIRECTORIES
    for d in os.listdir(BASE_DIR):
        if '_' in d:
            parts = d.split('_')
            try:
                qps = float(parts[-1])
                sched = "_".join(parts[:-1])
                
                run_dir = get_latest_run_dir(sched, parts[-1])
                if run_dir:
                    viol = calculate_violation(sched, parts[-1], run_dir)
                    if viol is not None:
                        if sched not in results: results[sched] = []
                        results[sched].append((qps, viol))
            except ValueError:
                continue

    # 2. ANALYZE CAPACITY
    final_capacities = {}
    
    # Ensure we check for all expected schedulers even if data is missing
    expected_schedulers = ['fcfs', 'edf', 'deadline_no_dynamic_chunking']

    for sched in expected_schedulers:
        if sched not in results:
            print(f"⚠️  Warning: No data found for {sched}")
            final_capacities[sched] = 0.0
            continue
            
        data = results[sched]
        data.sort(key=lambda x: x[0])
        
        # Print table
        print(f"--- {sched} ---")
        for qps, viol in data:
            status = "FAIL" if viol > VIOLATION_THRESHOLD else "PASS"
            print(f"{sched:<35} | {qps:<6.1f} | {viol:<11.2f}% | {status}")

        # Determine Capacity
        min_qps, min_viol = data[0]
        max_qps, max_viol = data[-1]
        capacity = 0.0
        
        if min_viol > VIOLATION_THRESHOLD:
            print(f"❌ {sched.upper()}: Fails at lowest QPS ({min_qps}). Capacity = 0.")
            capacity = 0.0
        elif max_viol <= VIOLATION_THRESHOLD:
            print(f"⚠️ {sched.upper()}: Passes at highest QPS ({max_qps}). Capacity >= {max_qps}.")
            capacity = max_qps
        else:
            for i in range(len(data)):
                if data[i][1] > VIOLATION_THRESHOLD:
                    capacity = data[i-1][0] if i > 0 else 0.0
                    print(f"✅ {sched.upper()}: Capacity found at {capacity} QPS.")
                    break
        
        final_capacities[sched] = capacity
        print("")

    # 3. PLOT
    plot_capacity_bars(final_capacities)

if __name__ == "__main__":
    main()