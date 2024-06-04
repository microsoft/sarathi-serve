#!/bin/bash

set -x

_scripts_dir=$(dirname "$(readlink -f "$0")")

# root dir is two levels up from the script's dir
ROOT_DIR=$(dirname $(dirname $_scripts_dir))

OUPUT_DIR=$ROOT_DIR/pipeline_parallel_latency_output
BATCH_SIZES=(8 16 32 64 128)

mkdir -p ${OUPUT_DIR}/tp8
mkdir -p ${OUPUT_DIR}/tp4_pp2


for batch_size in ${BATCH_SIZES[@]}; do
    python -m sarathi.benchmark.main \
        --model_name tiiuae/falcon-180B \
        --request_generator_provider synthetic \
        --synthetic_request_generator_length_provider trace \
        --synthetic_request_generator_interval_provider poisson \
        --trace_request_length_generator_max_tokens 8192 \
        --model_max_model_len 8192 \
        --trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
        --trace_request_length_generator_prefill_scale_factor 1 \
        --trace_request_length_generator_decode_scale_factor 1 \
        --synthetic_request_generator_num_requests 2048 \
        --replica_scheduler_provider sarathi \
        --replica_scheduler_max_batch_size ${batch_size} \
        --sarathi_scheduler_chunk_size 512 \
        --sarathi_scheduler_enable_rolling_prefills true \
        --model_tensor_parallel_degree 8 \
        --model_pipeline_parallel_degree 1 \
        --output_dir ${OUPUT_DIR}/tp8/bsz_${batch_size} \
        --poisson_request_interval_generator_qps 0.75 \
        --time_limit 1800 \
        --metrics_store_enable_op_level_metrics false \
        --metrics_store_enable_request_outputs false \
        --metrics_store_keep_individual_batch_metrics false \
        --write_chrome_trace false

    python -m sarathi.benchmark.main \
        --model_name tiiuae/falcon-180B \
        --request_generator_provider synthetic \
        --synthetic_request_generator_length_provider trace \
        --synthetic_request_generator_interval_provider poisson \
        --trace_request_length_generator_max_tokens 8192 \
        --model_max_model_len 8192 \
        --trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
        --trace_request_length_generator_prefill_scale_factor 1 \
        --trace_request_length_generator_decode_scale_factor 1 \
        --synthetic_request_generator_num_requests 2048 \
        --replica_scheduler_provider sarathi \
        --replica_scheduler_max_batch_size ${batch_size} \
        --sarathi_scheduler_chunk_size 512 \
        --sarathi_scheduler_enable_rolling_prefills true \
        --model_tensor_parallel_degree 4 \
        --model_pipeline_parallel_degree 2 \
        --output_dir ${OUPUT_DIR}/tp4_pp2/bsz_${batch_size} \
        --poisson_request_interval_generator_qps 0.75 \
        --time_limit 1800 \
        --metrics_store_enable_op_level_metrics false \
        --metrics_store_enable_request_outputs false \
        --metrics_store_keep_individual_batch_metrics false \
        --write_chrome_trace false
done
