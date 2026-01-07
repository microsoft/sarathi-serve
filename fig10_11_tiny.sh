#!/bin/bash

# ==============================================================================
# GPU ARGUMENT LOGIC
# ==============================================================================
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id1> <gpu_id2> ... [e.g., $0 0 2]"
    exit 1
fi

# Store the GPU IDs provided by the user
USER_GPUS=("$@")
NUM_USER_GPUS=${#USER_GPUS[@]}

echo ">>> Starting benchmark on GPUs: ${USER_GPUS[*]}"

# Configuration: High-fidelity engine, reduced search space
MODEL="meta-llama/Meta-Llama-3-8B"
PLOT_SCRIPT="paper_plot_scripts_tiny/fig10_11.py"
QPS_VALUES=(2.5 3.5 4.5)

# ==============================================================================
# WORKER GENERATION (Full paper config preserved)
# ==============================================================================
cat << 'EOF' > fig10_11_tiny_worker.sh
#!/bin/bash
SCHED_TYPE=$1; QPS=$2; GPU_ID=$3; MODEL=$4; NUM_REQS=$5; OUT_DIR=$6

mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/run.log"

CUDA_VISIBLE_DEVICES=$GPU_ID python -m sarathi.benchmark.main \
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
    --trace_request_length_generator_config_trace_file data/processed_traces/AzureLLMInferenceTrace_code_1week.csv \
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
chmod +x fig10_11_tiny_worker.sh

# ==============================================================================
# EXECUTION ENGINE (QPS Outer Loop, Scheduler Inner Loop)
# ==============================================================================
SCHEDULERS=("fcfs" "srpf" "edf" "deadline")
count=0

for qps in "${QPS_VALUES[@]}"; do
    for sched in "${SCHEDULERS[@]}"; do
        # Logic: 2.5 QPS gets 10k reqs, else 15k
        if [[ "$qps" == "2.5" ]]; then REQS=10000; else REQS=15000; fi
        
        # Select the GPU ID from the user-provided array
        gpu_idx=$((count % NUM_USER_GPUS))
        gpu_to_use=${USER_GPUS[$gpu_idx]}
        
        out="benchmark_output/fig_10_11_tiny/${sched}_${qps}"
        
        echo "Launching $sched @ $qps on GPU $gpu_to_use (Batch Index: $count)"
        ./fig10_11_tiny_worker.sh $sched $qps $gpu_to_use $MODEL $REQS $out &
        
        count=$((count + 1))
        
        # Parallelism limit: Wait after filling the available user GPUs
        if [[ $((count % NUM_USER_GPUS)) -eq 0 ]]; then
            echo ">>> Waiting for current batch to finish..."
            wait
        fi
    done
done
wait

python3 $PLOT_SCRIPT