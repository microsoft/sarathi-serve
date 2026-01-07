#!/bin/bash

# Define the QPS steps
QPS_VALUES=(2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0)

# Function to run the sweep on one GPU
run_gpu_sweep() {
    local gpu_id=$1
    local sched_type=$2
    
    echo ">>> Launching thread for $sched_type on GPU $gpu_id"
    
    for qps in "${QPS_VALUES[@]}"; do
        # Export GPU ID only for this command
        CUDA_VISIBLE_DEVICES=$gpu_id ./wrapper_exp.sh $sched_type $qps
        
        # Small sleep to prevent race conditions on file creation (optional but safe)
        sleep 2
    done
    
    echo ">>> Finished all runs for $sched_type on GPU $gpu_id"
}

# --- Launch Parallel Jobs ---

run_gpu_sweep 0 "fcfs" &
run_gpu_sweep 1 "srpf" &
run_gpu_sweep 2 "edf" &
run_gpu_sweep 3 "deadline" &

# Wait for all background jobs to finish
wait

echo "All experiments completed successfully."