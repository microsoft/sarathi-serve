#!/bin/bash
set -x
CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/figure-1/high_tail_latency_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 8192 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider poisson \
--poisson_request_interval_generator_qps 0.55 \
--synthetic_request_generator_num_requests 128 \
--synthetic_request_generator_length_provider trace \
--trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
--trace_request_length_generator_max_tokens 8192 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 256 \
--vllm_scheduler_max_tokens_in_batch 8192 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/figure-1/high_tail_latency_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 8192 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider poisson \
--poisson_request_interval_generator_qps 0.55 \
--synthetic_request_generator_num_requests 128 \
--synthetic_request_generator_length_provider trace \
--trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
--trace_request_length_generator_max_tokens 8192 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 256 \
--sarathi_scheduler_chunk_size 1536 \
--sarathi_scheduler_enable_rolling_prefills true \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/figure-1/high_tail_latency_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 8192 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider poisson \
--poisson_request_interval_generator_qps 0.7 \
--synthetic_request_generator_num_requests 128 \
--synthetic_request_generator_length_provider trace \
--trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
--trace_request_length_generator_max_tokens 8192 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 256 \
--vllm_scheduler_max_tokens_in_batch 8192 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/figure-1/high_tail_latency_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 8192 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider poisson \
--poisson_request_interval_generator_qps 0.7 \
--synthetic_request_generator_num_requests 128 \
--synthetic_request_generator_length_provider trace \
--trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
--trace_request_length_generator_max_tokens 8192 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 256 \
--sarathi_scheduler_chunk_size 1536 \
--sarathi_scheduler_enable_rolling_prefills true \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/figure-1/high_tail_latency_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 8192 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider poisson \
--poisson_request_interval_generator_qps 1.0 \
--synthetic_request_generator_num_requests 128 \
--synthetic_request_generator_length_provider trace \
--trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
--trace_request_length_generator_max_tokens 8192 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 256 \
--vllm_scheduler_max_tokens_in_batch 8192 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/llm-batching/osdi-experiments/figure-1/high_tail_latency_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 8192 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider poisson \
--poisson_request_interval_generator_qps 1.0 \
--synthetic_request_generator_num_requests 128 \
--synthetic_request_generator_length_provider trace \
--trace_request_length_generator_trace_file ./data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv \
--trace_request_length_generator_max_tokens 8192 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 256 \
--sarathi_scheduler_chunk_size 1536 \
--sarathi_scheduler_enable_rolling_prefills true \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

