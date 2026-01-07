import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# ==================================================================================
# PART 1: CONFIGURATION (Updated for Tiny Run)
# ==================================================================================
OUTPUT_DIR = "paper_plots_tiny/fig10_11"
BASE_DIR = "benchmark_output/fig_10_11_tiny" # Pointing to the tiny results

SLO_THRESHOLDS = {0: 6.0, 1: 600.0, 2: 1800.0}
LONG_REQUEST_THRESH = 6200

# Mapping Directory -> Legend Name
SCHEME_MAPPING = {
    'fcfs': 'Sarathi-FCFS',
    'srpf': 'Sarathi-SRPF',
    'edf': 'Sarathi-EDF',
    'deadline': 'Niyama'
}
ORDERED_SCHEMES = ['Sarathi-FCFS', 'Sarathi-SRPF', 'Sarathi-EDF', 'Niyama']

# ==================================================================================
# PART 2: DATA EXTRACTION
# ==================================================================================

def process_single_run(sched_type, qps, folder_path):
    # Find sequence_metrics.csv recursively in the tiny output subdirs
    csv_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder_path) 
                 for f in filenames if f == 'sequence_metrics.csv']
    
    if not csv_files: return None
    df = pd.read_csv(csv_files[-1])
    
    df['request_tier'] = df.get('request_tier', 2).replace(-1, 2)
    stats = {'Scheme': SCHEME_MAPPING.get(sched_type, sched_type), 'QPS': float(qps)}

    for tier in [0, 1, 2]:
        tier_df = df[df['request_tier'] == tier]
        count = len(tier_df)
        metric_col = 'prefill_e2e_time' if tier == 0 else 'request_e2e_time'
        slo_val = SLO_THRESHOLDS[tier]
        
        if count == 0:
            for p in ['p50', 'p95']: stats[f'T{tier}_{p}'] = np.nan
            stats[f'T{tier}_Viol%'] = 0.0
            stats[f'T{tier}_Short_Viol%'] = 0.0
            stats[f'T{tier}_Long_Viol%'] = 0.0
            continue

        # Latency
        stats[f'T{tier}_p50'] = np.percentile(tier_df[metric_col], 50)
        stats[f'T{tier}_p95'] = np.percentile(tier_df[metric_col], 95)

        # Violations
        stats[f'T{tier}_Viol%'] = (len(tier_df[tier_df[metric_col] > slo_val]) / count) * 100
        
        # Short vs Long
        s_df = tier_df[tier_df['request_num_prefill_tokens'] <= LONG_REQUEST_THRESH]
        l_df = tier_df[tier_df['request_num_prefill_tokens'] > LONG_REQUEST_THRESH]
        
        stats[f'T{tier}_Short_Viol%'] = (len(s_df[s_df[metric_col] > slo_val]) / len(s_df) * 100) if not s_df.empty else 0.0
        stats[f'T{tier}_Long_Viol%'] = (len(l_df[l_df[metric_col] > slo_val]) / len(l_df) * 100) if not l_df.empty else 0.0

    return stats

def aggregate_all_data():
    all_data = []
    if not os.path.exists(BASE_DIR):
        print(f"Error: {BASE_DIR} not found."); sys.exit(1)

    for d in os.listdir(BASE_DIR):
        if '_' in d:
            sched, qps = d.rsplit('_', 1)
            row = process_single_run(sched, qps, os.path.join(BASE_DIR, d))
            if row: all_data.append(row)
    
    df = pd.DataFrame(all_data)
    # Calculate Aggregates
    df['Overall_Viol'] = (df['T0_Viol%'] + df['T1_Viol%'] + df['T2_Viol%']) / 3
    df['Short_Viol'] = (df['T0_Short_Viol%'] + df['T1_Short_Viol%'] + df['T2_Short_Viol%']) / 3
    df['Long_Viol'] = (df['T0_Long_Viol%'] + df['T1_Long_Viol%'] + df['T2_Long_Viol%']) / 3
    return df

# ==================================================================================
# PART 3: RESTORED ORIGINAL PLOTTING
# ==================================================================================

plt.rcParams.update({'font.size': 18, 'font.family': 'sans-serif'})
COLORS = {'Sarathi-FCFS': '#d62728', 'Sarathi-SRPF': '#ff7f0e', 'Sarathi-EDF': '#2ca02c', 'Niyama': '#1f77b4'}
MARKERS = {'Sarathi-FCFS': 'o', 'Sarathi-SRPF': 's', 'Sarathi-EDF': '^', 'Niyama': 'D'}

def plot_latency(df):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    metrics = [['T0_p50', 'T1_p50', 'T2_p50'], ['T0_p95', 'T1_p95', 'T2_p95']]
    titles = [['(a) QoS 1 p50', '(b) QoS 2 p50', '(c) QoS 3 p50'], ['(d) QoS 1 p95', '(e) QoS 2 p95', '(f) QoS 3 p95']]

    for r in range(2):
        for c in range(3):
            ax = axs[r, c]
            for scheme in ORDERED_SCHEMES:
                data = df[df['Scheme'] == scheme].sort_values('QPS')
                if not data.empty:
                    ax.plot(data['QPS'], data[metrics[r][c]], label=scheme, color=COLORS[scheme], marker=MARKERS[scheme], linewidth=2.5)
            
            ax.axhline(y=SLO_THRESHOLDS[c], color='black', linestyle=':', label='SLO')
            ax.set_yscale('log')
            ax.set_title(titles[r][c], fontweight='bold')
            ax.set_xlabel('Load (QPS)')
            ax.set_ylabel('Latency (s)')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "fig10_latency.pdf"))

def plot_violations(df):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    metrics = [['Overall_Viol', 'Short_Viol', 'Long_Viol'], ['T0_Viol%', 'T1_Viol%', 'T2_Viol%']]
    titles = [['(a) Overall', '(b) Short', '(c) Long'], ['(d) QoS 1', '(e) QoS 2', '(f) QoS 3']]

    for r in range(2):
        for c in range(3):
            ax = axs[r, c]
            for scheme in ORDERED_SCHEMES:
                data = df[df['Scheme'] == scheme].sort_values('QPS')
                if not data.empty:
                    ax.plot(data['QPS'], data[metrics[r][c]], label=scheme, color=COLORS[scheme], marker=MARKERS[scheme], linewidth=2.5)
            ax.set_title(titles[r][c], fontweight='bold')
            ax.set_xlabel('Load (QPS)')
            ax.set_ylabel('Violation (%)')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "fig11_violations.pdf"))

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_df = aggregate_all_data()
    if master_df.empty:
        print("No data found to plot."); sys.exit(1)
    
    print("Generating High-Fidelity Paper Figures (Tiny Subset)...")
    plot_latency(master_df)
    plot_violations(master_df)
    print(f"Success. Files saved in {OUTPUT_DIR}")