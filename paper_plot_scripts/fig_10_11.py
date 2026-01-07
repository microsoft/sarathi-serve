import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# ==================================================================================
# PART 1: CONFIGURATION & MAPPING FOR FIGS 10 & 11
# ==================================================================================

# Specific Output Directory for these figures
OUTPUT_DIR = "plots_fig_10_11"

# Root benchmark directory
BASE_DIR = "benchmark_output"

# SLO Configurations
SLO_THRESHOLDS = {
    0: 6.0,    # Tier 0: 6s (Prefill E2E)
    1: 600.0,  # Tier 1: 600s (Request E2E)
    2: 1800.0  # Tier 2: 1800s (Request E2E)
}

# Threshold for Short vs Long requests (Prompt Length)
LONG_REQUEST_THRESH = 6200

# Mapping from Directory Name -> Paper Scheme Name
SCHEME_MAPPING = {
    'fcfs': 'fcfs_150k',
    'srpf': 'srpf_90k',
    'edf': 'EDF_QPS',
    'deadline': 'niyama'
}

# List of schemes to process (order matters for plotting legend)
ORDERED_SCHEMES = ['fcfs_150k', 'srpf_90k', 'EDF_QPS', 'niyama']

# ==================================================================================
# PART 2: DATA EXTRACTION & PROCESSING
# ==================================================================================

def get_latest_run_dir(sched_type, qps):
    """Finds the most recent timestamped run for a given scheduler and QPS."""
    base_path = os.path.join(BASE_DIR, f"{sched_type}_{qps}")
    if not os.path.exists(base_path):
        return None
    
    # Get all subdirectories (timestamped runs)
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) 
               if os.path.isdir(os.path.join(base_path, d))]
    
    if not subdirs:
        return None
        
    # Sort by creation time (latest last)
    return max(subdirs, key=os.path.getmtime)

def process_single_run(sched_type, qps, run_dir):
    """Parses sequence_metrics.csv and computes all stats for one run."""
    csv_path = os.path.join(run_dir, "replica_0", "sequence_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: No metrics found for {sched_type} @ {qps} ({csv_path})")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None
    
    # --- Preprocessing ---
    # Ensure request_tier exists. If -1 or missing, map to Tier 2 (Lowest Priority)
    if 'request_tier' not in df.columns:
        df['request_tier'] = 2
    else:
        df['request_tier'] = df['request_tier'].replace(-1, 2)
        
    # --- Calculate Stats per Tier ---
    stats = {
        'Scheme': SCHEME_MAPPING.get(sched_type, sched_type),
        'QPS': float(qps)
    }

    total_requests = 0
    total_violations = 0
    
    # Track summed violations for Short/Long global calc
    sum_short_viol_pct = 0
    sum_long_viol_pct = 0
    
    for tier in [0, 1, 2]:
        tier_df = df[df['request_tier'] == tier]
        count = len(tier_df)
        
        # Define metric column based on Tier
        metric_col = 'prefill_e2e_time' if tier == 0 else 'request_e2e_time'
        slo_val = SLO_THRESHOLDS[tier]
        
        if count == 0:
            # Fill with NaN if no data for this tier
            for p in ['p50', 'p90', 'p95', 'p99']:
                stats[f'T{tier}_{p}'] = np.nan
            stats[f'T{tier}_Viol%'] = 0.0
            stats[f'T{tier}_Short_Viol%'] = 0.0
            stats[f'T{tier}_Long_Viol%'] = 0.0
            continue

        # 1. Latency Percentiles
        stats[f'T{tier}_p50'] = np.percentile(tier_df[metric_col], 50)
        stats[f'T{tier}_p90'] = np.percentile(tier_df[metric_col], 90)
        stats[f'T{tier}_p95'] = np.percentile(tier_df[metric_col], 95)
        stats[f'T{tier}_p99'] = np.percentile(tier_df[metric_col], 99)

        # 2. Overall Violations for this Tier
        violations = tier_df[tier_df[metric_col] > slo_val]
        viol_count = len(violations)
        stats[f'T{tier}_Viol%'] = (viol_count / count) * 100
        
        total_requests += count
        total_violations += viol_count

        # 3. Short vs Long Violations for this Tier
        # Split original tier DF
        short_df = tier_df[tier_df['request_num_prefill_tokens'] <= LONG_REQUEST_THRESH]
        long_df = tier_df[tier_df['request_num_prefill_tokens'] > LONG_REQUEST_THRESH]
        
        # Compute Short Violations
        if len(short_df) > 0:
            short_viols = short_df[short_df[metric_col] > slo_val]
            s_val = (len(short_viols) / len(short_df)) * 100
        else:
            s_val = 0.0
        stats[f'T{tier}_Short_Viol%'] = s_val
        sum_short_viol_pct += s_val

        # Compute Long Violations
        if len(long_df) > 0:
            long_viols = long_df[long_df[metric_col] > slo_val]
            l_val = (len(long_viols) / len(long_df)) * 100
        else:
            l_val = 0.0
        stats[f'T{tier}_Long_Viol%'] = l_val
        sum_long_viol_pct += l_val

    # --- Aggregated Stats ---
    # Overall Violation % (Global requests)
    
    return stats

def aggregate_all_data():
    """Scans all dirs and builds the master DataFrame."""
    all_data = []
    
    # We scan the directory to find available QPS
    available_configs = []
    
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory '{BASE_DIR}' does not exist.")
        sys.exit(1)

    for d in os.listdir(BASE_DIR):
        if '_' in d:
            parts = d.split('_')
            # Assuming format: sched_qps
            if len(parts) >= 2:
                # Handle cases like "fcfs_2.5"
                qps_str = parts[-1]
                sched = "_".join(parts[:-1])
                
                if sched in SCHEME_MAPPING:
                    available_configs.append((sched, qps_str))

    print(f"Found {len(available_configs)} experiment configurations.")

    for sched, qps in available_configs:
        run_dir = get_latest_run_dir(sched, qps)
        if run_dir:
            print(f"Processing {sched} @ {qps} -> {run_dir}")
            row = process_single_run(sched, qps, run_dir)
            if row:
                all_data.append(row)
    
    if not all_data:
        print("No valid data found!")
        sys.exit(1)

    df = pd.DataFrame(all_data)
    return df.sort_values(by=['Scheme', 'QPS'])

# ==================================================================================
# PART 3: PLOTTING LOGIC FOR FIGS 10 & 11
# ==================================================================================

# Matplotlib styling
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif'})

# Plotting Constants
sysname = 'QoServe'
slo = [6, 600, 1800]
slo_color = "#0A0A0A"
slo_linestyle = ':'

labels = {
    'fcfs_150k': 'Sarathi-FCFS',
    'srpf_90k': 'Sarathi-SRPF',
    'EDF_QPS': 'Sarathi-EDF',
    'niyama': sysname
}

colors = {
    'fcfs_150k': '#d62728', 
    'srpf_90k': '#ff7f0e',  
    'EDF_QPS': '#2ca02c',   
    'niyama': '#1f77b4'     
}

markers = {
    'fcfs_150k': 'o', 
    'srpf_90k': 's', 
    'EDF_QPS': '^', 
    'niyama': 'D'   
}

linestyles = {
    'fcfs_150k': '-', 
    'srpf_90k': '--', 
    'EDF_QPS': '-.', 
    'niyama': '-'   
}

scheme_dfs = {}

def preprocess_for_plotting(df):
    """Calculates the aggregate columns expected by the plotting functions."""
    # Logic: Average of the percentages across tiers
    df['Overall_Viol'] = (df['T0_Viol%'] + df['T1_Viol%'] + df['T2_Viol%']) / 3
    df['Short_Viol'] = (df['T0_Short_Viol%'] + df['T1_Short_Viol%'] + df['T2_Short_Viol%']) / 3
    df['Long_Viol'] = (df['T0_Long_Viol%'] + df['T1_Long_Viol%'] + df['T2_Long_Viol%']) / 3
    
    # Split into scheme-specific DataFrames
    for scheme in ORDERED_SCHEMES:
        scheme_dfs[scheme] = df[df['Scheme'] == scheme].sort_values(by='QPS')


def plot_latency_fig10():
    """Generates Figure 10: Latency (p50 and p95) for all Tiers."""
    print("Generating Figure 10 (Latency)...")
    
    # 2 rows x 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(20, 8))
    
    # Row 1: p50
    metrics_p50 = ['T0_p50', 'T1_p50', 'T2_p50']
    titles_p50 = ['(a) QoS 1 (p50)', '(b) QoS 2 (p50)', '(c) QoS 3 (p50)']
    
    # Row 2: p95
    metrics_p95 = ['T0_p95', 'T1_p95', 'T2_p95']
    titles_p95 = ['(d) QoS 1 (p95)', '(e) QoS 2 (p95)', '(f) QoS 3 (p95)']

    # Loop for columns (Tiers)
    for col in range(3):
        # --- Top Row (p50) ---
        ax = axs[0, col]
        metric = metrics_p50[col]
        
        for scheme in ORDERED_SCHEMES:
            if scheme not in scheme_dfs or scheme_dfs[scheme].empty: continue
            data = scheme_dfs[scheme]
            ax.plot(data['QPS'], data[metric], label=labels[scheme], 
                    color=colors[scheme], marker=markers[scheme], 
                    linestyle=linestyles[scheme], linewidth=2.5, markersize=7)
            
        ax.axhline(y=slo[col], color=slo_color, linestyle=slo_linestyle, label='SLO', linewidth=3)
        ax.grid(axis='y', linestyle=slo_linestyle)
        ax.set_yscale('log')
        ax.set_xlabel('Load (QPS)', fontweight='bold', fontsize=18)
        
        ylabel = 'TTFT (s)' if col == 0 else 'TTLT (s)'
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=18)
        ax.set_title(titles_p50[col], fontweight='bold', fontsize=18)

        # --- Bottom Row (p95) ---
        ax = axs[1, col]
        metric = metrics_p95[col]
        
        for scheme in ORDERED_SCHEMES:
            if scheme not in scheme_dfs or scheme_dfs[scheme].empty: continue
            data = scheme_dfs[scheme]
            ax.plot(data['QPS'], data[metric], label=labels[scheme], 
                    color=colors[scheme], marker=markers[scheme], 
                    linestyle=linestyles[scheme], linewidth=2.5, markersize=7)

        ax.axhline(y=slo[col], color=slo_color, linestyle=slo_linestyle, label='SLO', linewidth=3)
        ax.grid(axis='y', linestyle=slo_linestyle)
        ax.set_yscale('log')
        ax.set_xlabel('Load (QPS)', fontweight='bold', fontsize=18)
        
        ylabel = 'TTFT (s)' if col == 0 else 'TTLT (s)'
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=18)
        ax.set_title(titles_p95[col], fontweight='bold', fontsize=18)

    # Legend
    handles, leg_label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_label, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=True, fontsize=18)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.8, hspace=0.4)
    
    save_path = os.path.join(OUTPUT_DIR, "fig_10.pdf")
    plt.savefig(save_path)
    print(f"Saved {save_path}")


def plot_violations_fig11():
    """Generates Figure 11: Deadline Violations."""
    print("Generating Figure 11 (Violations)...")

    # 2 rows x 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(20, 8))

    # --- Row 1: Overall, Short, Long ---
    metrics_r1 = ['Overall_Viol', 'Short_Viol', 'Long_Viol']
    titles_r1 = ['(a) Overall', '(b) Short', '(c) Long']

    for col in range(3):
        ax = axs[0, col]
        metric = metrics_r1[col]
        for scheme in ORDERED_SCHEMES:
            if scheme not in scheme_dfs or scheme_dfs[scheme].empty: continue
            data = scheme_dfs[scheme]
            ax.plot(data['QPS'], data[metric], label=labels[scheme], 
                    color=colors[scheme], marker=markers[scheme], 
                    linestyle=linestyles[scheme], linewidth=2.5, markersize=7)
        ax.grid(axis='y', linestyle=slo_linestyle)
        ax.set_xlabel('Load (QPS)', fontweight='bold', fontsize=18)
        ax.set_ylabel('Violation (%)', fontweight='bold', fontsize=18)
        ax.set_title(titles_r1[col], fontweight='bold', fontsize=18)

    # --- Row 2: QoS Specific Violations ---
    metrics_r2 = ['T0_Viol%', 'T1_Viol%', 'T2_Viol%']
    titles_r2 = ['(d) QoS 1', '(e) QoS 2', '(f) QoS 3']

    for col in range(3):
        ax = axs[1, col]
        metric = metrics_r2[col]
        for scheme in ORDERED_SCHEMES:
            if scheme not in scheme_dfs or scheme_dfs[scheme].empty: continue
            data = scheme_dfs[scheme]
            ax.plot(data['QPS'], data[metric], label=labels[scheme], 
                    color=colors[scheme], marker=markers[scheme], 
                    linestyle=linestyles[scheme], linewidth=2.5, markersize=7)
        ax.grid(axis='y', linestyle=slo_linestyle)
        ax.set_xlabel('Load (QPS)', fontweight='bold', fontsize=18)
        ax.set_ylabel(f'QoS {col+1} Violation (%)', fontweight='bold', fontsize=18)
        ax.set_title(titles_r2[col], fontweight='bold', fontsize=18)

    # Legend
    handles, leg_label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_label, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True, fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.8, hspace=0.4)
    
    save_path = os.path.join(OUTPUT_DIR, "fig_11.pdf")
    plt.savefig(save_path)
    print(f"Saved {save_path}")

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

if __name__ == "__main__":
    print(">>> Starting Data Aggregation...")
    df = aggregate_all_data()
    
    print(">>> Processing Data for Plots...")
    # Calculate stats columns before anything else
    preprocess_for_plotting(df)

    # Create specific output folder for Figs 10 & 11
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    # Save intermediate CSV for debugging
    csv_save_path = os.path.join(OUTPUT_DIR, "fig_10_11_consolidated_data.csv")
    df.to_csv(csv_save_path, index=False)
    print(f">>> Data aggregated. Saved to {csv_save_path}")
    
    print(">>> Generating Figures 10 and 11...")
    plot_latency_fig10()
    plot_violations_fig11()
    
    print(f">>> Done! Check the '{OUTPUT_DIR}' directory.")