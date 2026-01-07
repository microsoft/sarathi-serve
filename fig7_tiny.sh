#!/bin/bash

# ==============================================================================
# GPU ARGUMENT LOGIC
# ==============================================================================
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id1> <gpu_id2> ... [e.g., $0 0 2]"
    exit 1
fi

# Store the GPU IDs provided by the user (e.g., 0 2)
USER_GPUS=("$@")
NUM_USER_GPUS=${#USER_GPUS[@]}

echo ">>> Starting Fig 7 Tiny benchmark on GPUs: ${USER_GPUS[*]}"

# Configuration: Same as paper, just specific QPS points
MODEL="meta-llama/Meta-Llama-3-8B"
PLOT_SCRIPT="paper_plot_scripts_tiny/fig7.py"

# QPS selections for Tiny run
AZCODE_RUNS=("fcfs 1.9" "edf 2.9" "deadline 4.1")
AZCONV_RUNS=("fcfs 2.9" "edf 3.9" "deadline 4.7")
SHAREGPT_RUNS=("fcfs 1.3" "edf 2.1" "deadline 2.3")

# ==============================================================================
# WORKER GENERATION (Full paper config preserved)
# ==============================================================================
cat << 'EOF' > fig7_tiny_worker.sh
#!/bin/bash
SCHED_TYPE=$1; QPS=$2; GPU_ID=$3; MODEL=$4; TRACE_FILE=$5; NUM_REQS=$6; OUT_DIR=$7; THRESHOLD=$8

mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/run.log"

CUDA_VISIBLE_DEVICES=$GPU_ID python -m sarathi.benchmark.main \
    --output_dir $OUT_DIR \
    --scheduler_config_type 'DEADLINE' \
    --interval_generator_config_type 'POISSON' \
    --poisson_request_interval_generator_config_qps $QPS \
    --synthetic_request_generator_config_num_requests $NUM_REQS \
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
    if grep -q "decode_completion_time_series" "$LOG_FILE"; then break; fi
    if ! kill -0 $PY_PID 2>/dev/null; then break; fi
    sleep 2
done
kill -9 $PY_PID 2>/dev/null
wait $PY_PID 2>/dev/null
EOF
chmod +x fig7_tiny_worker.sh

# ==============================================================================
# EXECUTION ENGINE (Sequential Trace batches, parallel GPUs within batch)
# ==============================================================================
run_tiny_batch() {
    local NAME=$1; local FILE=$2; local REQS=$3; local THRESH=$4; shift 4
    local RUNS=("$@")
    echo ">>> Running Workload: $NAME (Threshold: $THRESH)"
    
    local count=0
    for i in "${!RUNS[@]}"; do
        set -- ${RUNS[$i]}
        local sched=$1; local qps=$2
        
        # KEY MODIFICATION: Select GPU from user-provided array
        local gpu_idx=$((count % NUM_USER_GPUS))
        local gpu_to_use=${USER_GPUS[$gpu_idx]}
        
        local out="benchmark_output/fig7_tiny/${NAME}/${sched}_${qps}"
        
        ./fig7_tiny_worker.sh $sched $qps $gpu_to_use $MODEL $FILE $REQS $out $THRESH &
        
        count=$((count + 1))
        
        # Parallelism limit: Wait after filling the available user GPUs
        if [[ $((count % NUM_USER_GPUS)) -eq 0 ]]; then
            wait
        fi
    done
    wait
}

run_tiny_batch "azcode" "data/processed_traces/AzureLLMInferenceTrace_code_1week.csv" 9000 0.05 "${AZCODE_RUNS[@]}"
run_tiny_batch "azconv" "data/processed_traces/AzureLLMInferenceTrace_conv_1week.csv" 10000 0.05 "${AZCONV_RUNS[@]}"
run_tiny_batch "sharegpt" "data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv" 4800 0.055 "${SHAREGPT_RUNS[@]}"

python3 $PLOT_SCRIPT