#!/bin/bash

# ==============================================================================
# CONFIGURATION AREA: Adjust QPS and Request Counts here
# ==============================================================================

# 1. Azure Code Trace
AZCODE_NUM_REQS=39000
AZCODE_QUEUES=(
    "fcfs 1.45" "edf 2.9" "deadline 3.55" # GPU 0
    "fcfs 1.6"  "edf 2.75" "deadline 3.7"  # GPU 1
    "fcfs 1.75" "edf 2.6"  "deadline 3.85" # GPU 2
    "fcfs 1.9"  "edf 2.45" "deadline 4.0"  # GPU 3
)

# 2. Azure Conv Trace
AZCONV_NUM_REQS=42000
AZCONV_QUEUES=(
    "fcfs 2.5" "edf 3.8" "deadline 4.0" # GPU 0
    "fcfs 2.7" "edf 3.6" "deadline 4.2" # GPU 1
    "fcfs 2.9" "edf 3.4" "deadline 4.4" # GPU 2
    "fcfs 3.1" "edf 3.2" "deadline 4.6" # GPU 3
)

# 3. ShareGPT Trace
SHAREGPT_NUM_REQS=18000
SHAREGPT_QUEUES=(
    "fcfs 1.1" "edf 2.1" "deadline 1.7" # GPU 0
    "fcfs 1.3" "edf 1.9" "deadline 1.9" # GPU 1
    "fcfs 1.5" "edf 1.7" "deadline 2.1" # GPU 2
    "fcfs 1.7" "edf 1.5" "deadline 2.3" # GPU 3
)

# Common settings
MODEL="meta-llama/Meta-Llama-3-8B"
PLOT_SCRIPT="paper_plot_scripts/fig7_cap.py"

# ==============================================================================
# STEP 1: Script Logic & Worker Generation
# ==============================================================================

usage() {
    echo "Usage: $0 [--traces azcode,azconv,sharegpt]"
    echo "Default: Runs all traces."
    exit 1
}

# Parse selected traces
SELECTED_TRACES="azcode,azconv,sharegpt"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --traces) SELECTED_TRACES="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

echo "Creating fig7_wrapper.sh..."
cat << 'EOF' > fig7_wrapper.sh
#!/bin/bash
# Internal worker script
SCHED_TYPE=$1
QPS=$2
GPU_ID=$3
MODEL=$4
TRACE_FILE=$5
NUM_REQS=$6
OUT_DIR=$7
THRESHOLD=${8:-0.05} # Default to 0.05 if not provided

LOG_FILE="${OUT_DIR}/run.log"
mkdir -p "$OUT_DIR"

echo "=== [GPU $GPU_ID] $SCHED_TYPE @ $QPS QPS (Reqs: $NUM_REQS) ==="

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
    --deadline_scheduler_config_execution_threshold $THRESHOLD \
    --deadline_scheduler_config_execution_threshold_batched 0.4 \
    --model_config_max_model_len 260000 \
    --trace_request_length_generator_config_trace_file $TRACE_FILE \
    > "$LOG_FILE" 2>&1 &

PY_PID=$!
while true; do
    if grep -q "decode_completion_time_series" "$LOG_FILE"; then
        echo "✅ [GPU $GPU_ID] Trigger detected."
        break
    fi
    if ! kill -0 $PY_PID 2>/dev/null; then
        echo "❌ [GPU $GPU_ID] Process died! See $LOG_FILE"
        break
    fi
    sleep 2
done
kill -9 $PY_PID 2>/dev/null
wait $PY_PID 2>/dev/null
EOF
chmod +x fig7_wrapper.sh

# ==============================================================================
# STEP 2: Execution Engine
# ==============================================================================

run_trace_batch() {
    local TRACE_NAME=$1
    local TRACE_FILE=$2
    local NUM_REQS=$3
    local THRESHOLD=$4
    shift 4
    local QUEUE=("$@")

    echo ">>> Starting Trace: $TRACE_NAME"
    
    # Process 4 GPUs in parallel
    for gpu in {0..3}; do
        (
            # Extract 3 items per GPU from the flat QUEUE array
            local start_idx=$((gpu * 3))
            for i in {0..2}; do
                local item=${QUEUE[$((start_idx + i))]}
                set -- $item
                local sched=$1
                local qps=$2
                local out="benchmark_output/fig7/llama3_8b_${TRACE_NAME}/${sched}_${qps}"
                
                CUDA_VISIBLE_DEVICES=$gpu ./fig7_wrapper.sh $sched $qps $gpu $MODEL $TRACE_FILE $NUM_REQS $out $THRESHOLD
                sleep 2
            done
        ) &
    done
    wait
}

# Run selected traces sequentially
if [[ $SELECTED_TRACES == *"azcode"* ]]; then
    run_trace_batch "azcode" "data/processed_traces/AzureLLMInferenceTrace_code_1week.csv" $AZCODE_NUM_REQS 0.05 "${AZCODE_QUEUES[@]}"
fi

if [[ $SELECTED_TRACES == *"azconv"* ]]; then
    run_trace_batch "azconv" "data/processed_traces/AzureLLMInferenceTrace_conv_1week.csv" $AZCONV_NUM_REQS 0.05 "${AZCONV_QUEUES[@]}"
fi

if [[ $SELECTED_TRACES == *"sharegpt"* ]]; then
    run_trace_batch "sharegpt" "data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv" $SHAREGPT_NUM_REQS 0.055 "${SHAREGPT_QUEUES[@]}"
fi

# ==============================================================================
# STEP 3: Plotting
# ==============================================================================
if [ -f "$PLOT_SCRIPT" ]; then
    echo ">>> Running Plotting Script..."
    python "$PLOT_SCRIPT"
else
    echo "ERROR: Plotting script not found at $PLOT_SCRIPT"
fi

echo ">>> All reproductions complete."