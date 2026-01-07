#!/bin/bash

# ==============================================================================
# SARATHI-SERVE ZERO-FOOTPRINT TESTER
# ==============================================================================
# Usage: ./test_setup.sh
# Runs in foreground. Ctrl+C to stop. Cleans up automatically.

# Configuration
MODEL="meta-llama/Meta-Llama-3-8B"
TRACE_FILE="data/processed_traces/AzureLLMInferenceTrace_code_1week.csv"
TEMP_DIR="benchmark_output/temp_tester_$(date +%s)"
NUM_REQS=50

# --- CLEANUP FUNCTION ---
# This runs automatically when the script exits or you press Ctrl+C
cleanup() {
    echo ""
    echo ">>> Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# --- RUN EXPERIMENT ---
echo ">>> Starting Basic Sanity Check (50 Reqs)..."
echo ">>> Output Directory: $TEMP_DIR (Will be deleted automatically)"
mkdir -p "$TEMP_DIR"

# Run Python directly in foreground so you can see stdout/stderr
python -m sarathi.benchmark.main \
    --output_dir $TEMP_DIR \
    --scheduler_config_type 'DEADLINE' \
    --interval_generator_config_type 'POISSON' \
    --poisson_request_interval_generator_config_qps 2.0 \
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
    --deadline_scheduler_config_scheduler_type 'deadline' \
    --sarathi_scheduler_config_chunk_size 384 \
    --deadline_scheduler_config_execution_threshold 0.05 \
    --model_config_max_model_len 260000 \
    --trace_request_length_generator_config_trace_file $TRACE_FILE

# --- CHECK STATUS ---
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ BASIC TESTER RUN: PASSED"
else
    echo ""
    echo "❌ BASIC TESTER RUN: FAILED (Check output above)"
fi