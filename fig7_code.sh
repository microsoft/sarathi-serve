#!/bin/bash

# ==============================================================================
# STEP 1: Generate the Worker Script (fig7_wrapper.sh)
# ==============================================================================
echo "Creating fig7_wrapper.sh..."

cat << 'EOF' > fig7_wrapper.sh
#!/bin/bash

# Arguments
SCHED_TYPE=$1
QPS=$2
GPU_ID=$3

# Configuration for this Figure
MODEL="meta-llama/Meta-Llama-3-8B"
TRACE_FILE="data/processed_traces/AzureLLMInferenceTrace_code_1week.csv"

# Fixed Request Count for all schedulers
NUM_REQS=39000

# Output Directory Structure: benchmark_output/fig7/model_trace/sched_type_qps
BASE_OUT_DIR="benchmark_output/fig7/llama3_8b_azcode"
OUT_DIR="${BASE_OUT_DIR}/${SCHED_TYPE}_${QPS}"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "$OUT_DIR"

echo "=== [GPU $GPU_ID] Starting: $SCHED_TYPE @ $QPS QPS (Reqs: $NUM_REQS) ==="

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
    --model_config_model $MODEL \
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
    --trace_request_length_generator_config_trace_file $TRACE_FILE \
    > "$LOG_FILE" 2>&1 &

# 2. Capture the Python Process ID
PY_PID=$!

# 3. Robust Monitor Loop
echo "   [GPU $GPU_ID] Monitoring PID $PY_PID..."

while true; do
    # Check if the trigger string exists in the log
    if grep -q "decode_completion_time_series" "$LOG_FILE"; then
        echo "✅ [GPU $GPU_ID] Trigger detected. Stopping..."
        break
    fi

    # Check if the Python process is still running
    if ! kill -0 $PY_PID 2>/dev/null; then
        echo "❌ [GPU $GPU_ID] Process $PY_PID died unexpectedly! Check $LOG_FILE for errors."
        break
    fi

    # Sleep briefly
    sleep 1
done

# 4. Cleanup
kill -9 $PY_PID 2>/dev/null
wait $PY_PID 2>/dev/null
EOF

chmod +x fig7_wrapper.sh


# ==============================================================================
# STEP 2: Define and Run the Batches (Zig-Zag Load Balanced)
# ==============================================================================

# Bucket for GPU 0 (Mixed Load)
queue_0=(
"fcfs 1.45"
"edf 2.9"
"deadline 3.55"
)

# Bucket for GPU 1 (Mixed Load)
queue_1=(
"fcfs 1.6"
"edf 2.75"
"deadline 3.7"
)

# Bucket for GPU 2 (Mixed Load)
queue_2=(
"fcfs 1.75"
"edf 2.6"
"deadline 3.85"
)

# Bucket for GPU 3 (Mixed Load)
queue_3=(
"fcfs 1.9"
"edf 2.45"
"deadline 4.0"
)


# Function to process a queue on a specific GPU
run_gpu_bucket() {
    local gpu_id=$1
    # Get the array variable name passed as the second argument
    local -n queue=$2
    
    echo ">>> Launching Bucket on GPU $gpu_id (Total items: ${#queue[@]})"
    
    for item in "${queue[@]}"; do
        # Split string "sched qps" into variables
        set -- $item
        local sched=$1
        local qps=$2
        
        # Run the wrapper
        CUDA_VISIBLE_DEVICES=$gpu_id ./fig7_wrapper.sh $sched $qps $gpu_id
        
        # Brief pause
        sleep 2
    done
    
    echo ">>> Finished Bucket on GPU $gpu_id"
}

echo ">>> Starting Figure 7 Capacity Search (12 experiments on 4 GPUs)..."

# Launch the 4 buckets in parallel
run_gpu_bucket 0 queue_0 &
run_gpu_bucket 1 queue_1 &
run_gpu_bucket 2 queue_2 &
run_gpu_bucket 3 queue_3 &

# Wait for everything
wait

echo ">>> Figure 7 experiments completed."

# ==============================================================================
# STEP 3: Generate Figure 7 Capacity Plots
# ==============================================================================

echo ">>> Running Plotting Script..."

PLOT_SCRIPT="paper_plot_scripts/fig7_cap.py"

if [ -f "$PLOT_SCRIPT" ]; then
    python "$PLOT_SCRIPT"
else
    echo "ERROR: Plotting script not found at $PLOT_SCRIPT"
    echo "Please ensure the file exists at that path."
fi

echo ">>> Figure 7 Reproduction Complete."