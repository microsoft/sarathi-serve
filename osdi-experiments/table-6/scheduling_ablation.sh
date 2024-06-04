#!/bin/bash

set -x

_scripts_dir=$(dirname "$(readlink -f "$0")")

# root dir is two levels up from the script's dir
ROOT_DIR=$(dirname $(dirname $_scripts_dir))

OUPUT_DIR=$ROOT_DIR/scheduling_ablation_output

mkdir -p ${OUPUT_DIR}/sharechat
mkdir -p ${OUPUT_DIR}/arxiv


python -m sarathi.benchmark.main \
    --model_name 01-ai/Yi-34B \
    --request_generator_provider synthetic \
    --synthetic_request_generator_length_provider trace \
    --synthetic_request_generator_interval_provider poisson \
    --trace_request_length_generator_max_tokens 8192 \
    --model_max_model_len 8192 \
    --trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
    --trace_request_length_generator_prefill_scale_factor 1 \
    --trace_request_length_generator_decode_scale_factor 1 \
    --synthetic_request_generator_num_requests 128 \
    --replica_scheduler_provider sarathi \
    --replica_scheduler_max_batch_size 128 \
    --sarathi_scheduler_chunk_size 8192 \
    --sarathi_scheduler_enable_rolling_prefills true \
    --model_tensor_parallel_degree 2 \
    --model_pipeline_parallel_degree 1 \
    --output_dir ${OUPUT_DIR}/sharechat/hybird_only \
    --poisson_request_interval_generator_qps 1 \
    --time_limit 1800 \
    --metrics_store_enable_op_level_metrics false \
    --metrics_store_enable_request_outputs false \
    --metrics_store_keep_individual_batch_metrics false \
    --write_chrome_trace false

python -m sarathi.benchmark.main \
    --model_name 01-ai/Yi-34B \
    --request_generator_provider synthetic \
    --synthetic_request_generator_length_provider trace \
    --synthetic_request_generator_interval_provider poisson \
    --trace_request_length_generator_max_tokens 16384 \
    --model_max_model_len 16384 \
    --trace_request_length_generator_trace_file ./data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv \
    --trace_request_length_generator_prefill_scale_factor 1 \
    --trace_request_length_generator_decode_scale_factor 1 \
    --synthetic_request_generator_num_requests 128 \
    --replica_scheduler_provider sarathi \
    --replica_scheduler_max_batch_size 128 \
    --sarathi_scheduler_chunk_size 16384 \
    --sarathi_scheduler_enable_rolling_prefills true \
    --model_tensor_parallel_degree 2 \
    --model_pipeline_parallel_degree 1 \
    --output_dir ${OUPUT_DIR}/arxiv/hybird_only \
    --poisson_request_interval_generator_qps 0.5 \
    --time_limit 1800 \
    --metrics_store_enable_op_level_metrics false \
    --metrics_store_enable_request_outputs false \
    --metrics_store_keep_individual_batch_metrics false \
    --write_chrome_trace false

python -m sarathi.benchmark.main \
    --model_name 01-ai/Yi-34B \
    --request_generator_provider synthetic \
    --synthetic_request_generator_length_provider trace \
    --synthetic_request_generator_interval_provider poisson \
    --trace_request_length_generator_max_tokens 8192 \
    --model_max_model_len 8192 \
    --trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
    --trace_request_length_generator_prefill_scale_factor 1 \
    --trace_request_length_generator_decode_scale_factor 1 \
    --synthetic_request_generator_num_requests 128 \
    --replica_scheduler_provider sarathi \
    --replica_scheduler_max_batch_size 128 \
    --sarathi_scheduler_chunk_size 512 \
    --sarathi_scheduler_enable_rolling_prefills true \
    --model_tensor_parallel_degree 2 \
    --model_pipeline_parallel_degree 1 \
    --output_dir ${OUPUT_DIR}/sharechat/combined \
    --poisson_request_interval_generator_qps 1 \
    --time_limit 1800 \
    --metrics_store_enable_op_level_metrics false \
    --metrics_store_enable_request_outputs false \
    --metrics_store_keep_individual_batch_metrics false \
    --write_chrome_trace false

python -m sarathi.benchmark.main \
    --model_name 01-ai/Yi-34B \
    --request_generator_provider synthetic \
    --synthetic_request_generator_length_provider trace \
    --synthetic_request_generator_interval_provider poisson \
    --trace_request_length_generator_max_tokens 16384 \
    --model_max_model_len 16384 \
    --trace_request_length_generator_trace_file ./data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv \
    --trace_request_length_generator_prefill_scale_factor 1 \
    --trace_request_length_generator_decode_scale_factor 1 \
    --synthetic_request_generator_num_requests 128 \
    --replica_scheduler_provider sarathi \
    --replica_scheduler_max_batch_size 128 \
    --sarathi_scheduler_chunk_size 512 \
    --sarathi_scheduler_enable_rolling_prefills true \
    --model_tensor_parallel_degree 2 \
    --model_pipeline_parallel_degree 1 \
    --output_dir ${OUPUT_DIR}/arxiv/combined \
    --poisson_request_interval_generator_qps 0.5 \
    --time_limit 1800 \
    --metrics_store_enable_op_level_metrics false \
    --metrics_store_enable_request_outputs false \
    --metrics_store_keep_individual_batch_metrics false \
    --write_chrome_trace false

python -m sarathi.benchmark.main \
    --model_name 01-ai/Yi-34B \
    --request_generator_provider synthetic \
    --synthetic_request_generator_length_provider trace \
    --synthetic_request_generator_interval_provider poisson \
    --trace_request_length_generator_max_tokens 8192 \
    --model_max_model_len 8192 \
    --trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
    --trace_request_length_generator_prefill_scale_factor 1 \
    --trace_request_length_generator_decode_scale_factor 1 \
    --synthetic_request_generator_num_requests 128 \
    --replica_scheduler_provider simple_chunking \
    --replica_scheduler_max_batch_size 128 \
    --sarathi_scheduler_chunk_size 512 \
    --sarathi_scheduler_enable_rolling_prefills true \
    --model_tensor_parallel_degree 2 \
    --model_pipeline_parallel_degree 1 \
    --output_dir ${OUPUT_DIR}/sharechat/chunking_only \
    --poisson_request_interval_generator_qps 1 \
    --time_limit 1800 \
    --metrics_store_enable_op_level_metrics false \
    --metrics_store_enable_request_outputs false \
    --metrics_store_keep_individual_batch_metrics false \
    --write_chrome_trace false

python -m sarathi.benchmark.main \
    --model_name 01-ai/Yi-34B \
    --request_generator_provider synthetic \
    --synthetic_request_generator_length_provider trace \
    --synthetic_request_generator_interval_provider poisson \
    --trace_request_length_generator_max_tokens 16384 \
    --model_max_model_len 16384 \
    --trace_request_length_generator_trace_file ./data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv \
    --trace_request_length_generator_prefill_scale_factor 1 \
    --trace_request_length_generator_decode_scale_factor 1 \
    --synthetic_request_generator_num_requests 128 \
    --replica_scheduler_provider simple_chunking \
    --replica_scheduler_max_batch_size 128 \
    --sarathi_scheduler_chunk_size 512 \
    --sarathi_scheduler_enable_rolling_prefills true \
    --model_tensor_parallel_degree 2 \
    --model_pipeline_parallel_degree 1 \
    --output_dir ${OUPUT_DIR}/arxiv/chunking_only \
    --poisson_request_interval_generator_qps 0.5 \
    --time_limit 1800 \
    --metrics_store_enable_op_level_metrics false \
    --metrics_store_enable_request_outputs false \
    --metrics_store_keep_individual_batch_metrics false \
    --write_chrome_trace false
