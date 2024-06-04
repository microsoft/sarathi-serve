#!/bin/bash
set -x
CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 2048 \
--uniform_request_length_generator_min_tokens 2048 \
--uniform_request_length_generator_prefill_to_decode_ratio 2047 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 512 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 2048 \
--uniform_request_length_generator_min_tokens 2048 \
--uniform_request_length_generator_prefill_to_decode_ratio 2047 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 1024 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 2048 \
--uniform_request_length_generator_min_tokens 2048 \
--uniform_request_length_generator_prefill_to_decode_ratio 2047 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 2048 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 2048 \
--uniform_request_length_generator_min_tokens 2048 \
--uniform_request_length_generator_prefill_to_decode_ratio 2047 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 16384 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 4096 \
--uniform_request_length_generator_min_tokens 4096 \
--uniform_request_length_generator_prefill_to_decode_ratio 4095 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 512 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 4096 \
--uniform_request_length_generator_min_tokens 4096 \
--uniform_request_length_generator_prefill_to_decode_ratio 4095 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 1024 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 4096 \
--uniform_request_length_generator_min_tokens 4096 \
--uniform_request_length_generator_prefill_to_decode_ratio 4095 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 2048 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 4096 \
--uniform_request_length_generator_min_tokens 4096 \
--uniform_request_length_generator_prefill_to_decode_ratio 4095 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 16384 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 8192 \
--uniform_request_length_generator_min_tokens 8192 \
--uniform_request_length_generator_prefill_to_decode_ratio 8191 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 512 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 8192 \
--uniform_request_length_generator_min_tokens 8192 \
--uniform_request_length_generator_prefill_to_decode_ratio 8191 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 1024 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 8192 \
--uniform_request_length_generator_min_tokens 8192 \
--uniform_request_length_generator_prefill_to_decode_ratio 8191 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 2048 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 8192 \
--uniform_request_length_generator_min_tokens 8192 \
--uniform_request_length_generator_prefill_to_decode_ratio 8191 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 16384 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 16384 \
--uniform_request_length_generator_min_tokens 16384 \
--uniform_request_length_generator_prefill_to_decode_ratio 16383 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 512 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 16384 \
--uniform_request_length_generator_min_tokens 16384 \
--uniform_request_length_generator_prefill_to_decode_ratio 16383 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 1024 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 16384 \
--uniform_request_length_generator_min_tokens 16384 \
--uniform_request_length_generator_prefill_to_decode_ratio 16383 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 2048 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-9a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 5 \
--uniform_request_length_generator_max_tokens 16384 \
--uniform_request_length_generator_min_tokens 16384 \
--uniform_request_length_generator_prefill_to_decode_ratio 16383 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 1 \
--sarathi_scheduler_chunk_size 16384 \
--sarathi_scheduler_enable_rolling_prefills false \
--sarathi_scheduler_enable_dynamic_chunking_schedule false
