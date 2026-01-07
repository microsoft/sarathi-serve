import os
import pandas as pd

BASE_DIR = "benchmark_output/fig7_tiny"
SLO = {0: 6.0, 1: 600.0, 2: 1800.0}

def analyze_tiers(csv_path):
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path)
        df['request_tier'] = df.get('request_tier', 2).replace(-1, 2)
        
        results = {}
        for tier in [0, 1, 2]:
            tier_df = df[df['request_tier'] == tier]
            if len(tier_df) == 0:
                results[tier] = "0.00%*" # No requests in this tier
                continue
            col = 'prefill_e2e_time' if tier == 0 else 'request_e2e_time'
            viol = (tier_df[col] > SLO[tier]).mean() * 100
            results[tier] = f"{viol:.2f}%"
        return results
    except Exception: return None

print(f"\n{'WORKLOAD':<12} | {'SCHEME':<12} | {'QPS':<6} | {'T0 VIOL':<8} | {'T1 VIOL':<8} | {'T2 VIOL':<8}")
print("-" * 75)

for workload in sorted(os.listdir(BASE_DIR)):
    w_path = os.path.join(BASE_DIR, workload)
    if not os.path.isdir(w_path): continue
    
    for run in sorted(os.listdir(w_path)):
        if '_' not in run: continue
        sched, qps = run.rsplit('_', 1)
        run_dir = os.path.join(w_path, run)
        
        # Find latest sequence_metrics.csv
        csv_path = None
        for root, dirs, files in os.walk(run_dir):
            if "sequence_metrics.csv" in files:
                csv_path = os.path.join(root, "sequence_metrics.csv")
                break
        
        if csv_path:
            stats = analyze_tiers(csv_path)
            if stats:
                print(f"{workload:<12} | {sched:<12} | {qps:<6} | {stats[0]:<8} | {stats[1]:<8} | {stats[2]:<8}")