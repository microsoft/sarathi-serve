#!/bin/bash

# ==============================================================================
# STEP 1: Generate the Worker Script (wrapper_exp.sh)
# ==============================================================================
echo "Creating wrapper_exp.sh..."

cat << 'EOF' > wrapper_exp.sh
#!/bin/bash

# Arguments
SCHED_TYPE=$1
QPS=$2

# --- Logic: Dynamic Request Count ---
# If QPS is 2.5 or 3.0, use 50k requests. Otherwise (3.5+), use 75k.
if [[ "$QPS" == "2.5" || "$QPS" == "3.0" ]]; then
    NUM_REQS=50000
else
    NUM_REQS=75000
fi

# Setup output paths
OUT_DIR="benchmark_output/${SCHED_TYPE}_${QPS}"
LOG_FILE="${OUT_DIR}/run.log"
mkdir -p "$OUT_DIR"

echo "=== [GPU $CUDA_VISIBLE_DEVICES] Starting: $SCHED_TYPE @ $QPS QPS (Reqs: $NUM_REQS) ==="

# 1. Start Python in the background
python -m sarathi.benchmark.main \
    --output_dir $OUT_DIR \
    --scheduler_config_type 'DEADLINE' \
    --interval_generator_config_type 'POISSON' \
    --poisson_request_interval_generator_config_qps $QPS \
    --synthetic_request_generator_config_num_requests $NUM_REQS  \
    --worker_config_attention_backend 'FLASHINFER' \
    --fixed_request_length_generator_config_decode_tokens 375 \
    --fixed_request_length_generator_config_prefill_tokens 3750 \
    --vllm_scheduler_config_max_num_seqs 1000 \
    --sarathi_scheduler_config_max_num_seqs 1000 \
    --sarathi_scheduler_config_chunk_schedule_max_tokens 8192 \
    --vllm_scheduler_config_max_batched_tokens 8192 \
    --time_limit 33000000000 \
    --model_config_load_format 'DUMMY' \
    --length_generator_config_type 'TRACE' \
    --request_generator_config_type 'SYNTHETIC' \
    --model_config_model 'meta-llama/Meta-Llama-3-8B' \
    --parallel_config_pipeline_parallel_size 1 \
    --parallel_config_tensor_parallel_size 1 \
    --worker_config_gpu_memory_utilization 0.9 \
    --metrics_config_keep_individual_batch_metrics \
    --metrics_config_enable_cpu_op_level_metrics \
    --deadline_scheduler_config_chunk_size 256 \
    --deadline_scheduler_config_scheduler_type $SCHED_TYPE \
    --sarathi_scheduler_config_chunk_size 384 \
    --deadline_scheduler_config_execution_threshold 0.05 \
    --deadline_scheduler_config_execution_threshold_batched 0.4 \
    --model_config_max_model_len 260000 \
    --trace_request_length_generator_config_trace_file data/processed_traces/AzureLLMInferenceTrace_code_1week.csv \
    > "$LOG_FILE" 2>&1 &

# 2. Capture the Python Process ID
PY_PID=$!

# 3. Robust Monitor Loop
# Checks if process is alive AND if trigger string appears.
echo "   [GPU $CUDA_VISIBLE_DEVICES] Monitoring PID $PY_PID..."

while true; do
    # Check if the trigger string exists in the log
    if grep -q "decode_completion_time_series" "$LOG_FILE"; then
        echo "✅ [GPU $CUDA_VISIBLE_DEVICES] Trigger detected. Stopping..."
        break
    fi

    # Check if the Python process is still running (kill -0 checks existence)
    if ! kill -0 $PY_PID 2>/dev/null; then
        echo "❌ [GPU $CUDA_VISIBLE_DEVICES] Process $PY_PID died unexpectedly! Check $LOG_FILE for errors."
        break
    fi

    # Sleep briefly to save CPU
    sleep 1
done

# 4. Cleanup: Kill the process if it's still alive (trigger case)
kill -9 $PY_PID 2>/dev/null

# 5. Wait briefly to clean up zombie process
wait $PY_PID 2>/dev/null
EOF

# Make the generated script executable
chmod +x wrapper_exp.sh


# ==============================================================================
# STEP 2: Run the Parallel Sweeps
# ==============================================================================

# Define the QPS steps
QPS_VALUES=(2.5 3.5 4.0 4.5 5.5)

# Function to run the sweep on one GPU
run_gpu_sweep() {
    local gpu_id=$1
    local sched_type=$2
    
    echo ">>> Launching thread for $sched_type on GPU $gpu_id"
    
    for qps in "${QPS_VALUES[@]}"; do
        # Export GPU ID only for this command
        CUDA_VISIBLE_DEVICES=$gpu_id ./wrapper_exp.sh $sched_type $qps
        
        # Small sleep to prevent race conditions on file creation
        sleep 2
    done
    
    echo ">>> Finished all runs for $sched_type on GPU $gpu_id"
}

echo ">>> Starting Parallel Experiments..."

# Launch Parallel Jobs
run_gpu_sweep 0 "fcfs" &
run_gpu_sweep 1 "srpf" &
run_gpu_sweep 2 "edf" &
run_gpu_sweep 3 "deadline" &

# Wait for all background jobs to finish
wait

echo ">>> All experiments completed successfully."


# ==============================================================================
# STEP 3: Generate Figures 10 and 11
# ==============================================================================

echo ">>> Running Plotting Script..."

# Uses the explicit path you requested: paper_plot_scripts/fig_10_11.py
PLOT_SCRIPT="paper_plot_scripts/fig_10_11.py"

if [ -f "$PLOT_SCRIPT" ]; then
    python "$PLOT_SCRIPT"
else
    echo "ERROR: Plotting script not found at $PLOT_SCRIPT"
    echo "Please ensure the file exists at that path."
fi

echo ">>> Reproduction of Fig 10 & 11 Complete."