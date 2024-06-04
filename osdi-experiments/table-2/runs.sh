#!/bin/bash
set -x
CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/table-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1040 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 8 \
--synthetic_request_generator_length_provider fixed \
--fixed_request_length_generator_prefill_tokens 1024 \
--fixed_request_length_generator_decode_tokens 16 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 4 \
--vllm_scheduler_max_tokens_in_batch 1040 \
--metrics_store_enable_op_level_metrics false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/table-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1040 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 8 \
--synthetic_request_generator_length_provider fixed \
--fixed_request_length_generator_prefill_tokens 1024 \
--fixed_request_length_generator_decode_tokens 16 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 4 \
--vllm_scheduler_max_tokens_in_batch 1040 \
--metrics_store_enable_op_level_metrics true

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/table-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1040 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 8 \
--synthetic_request_generator_length_provider fixed \
--fixed_request_length_generator_prefill_tokens 1024 \
--fixed_request_length_generator_decode_tokens 16 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 4 \
--sarathi_scheduler_chunk_size 1024 \
--sarathi_scheduler_enable_rolling_prefills true \
--sarathi_scheduler_enable_dynamic_chunking_schedule false \
--metrics_store_enable_op_level_metrics false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/table-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1040 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 8 \
--synthetic_request_generator_length_provider fixed \
--fixed_request_length_generator_prefill_tokens 1024 \
--fixed_request_length_generator_decode_tokens 16 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 4 \
--sarathi_scheduler_chunk_size 1024 \
--sarathi_scheduler_enable_rolling_prefills true \
--sarathi_scheduler_enable_dynamic_chunking_schedule false \
--metrics_store_enable_op_level_metrics true

