#!/bin/bash
set -x
CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-1a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 128 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_generation_stalls_experiments \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_arxiv-summarization_filtered_seq16384_static_num_reqs_128_vllm_chunk_size_None_batch_size_256_55e75be9 \
--synthetic_request_generator_length_provider trace \
--trace_request_length_generator_trace_file ./data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv \
--trace_request_length_generator_max_tokens 16384 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 256 \
--vllm_scheduler_max_tokens_in_batch 16384 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-1a/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16384 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 128 \
--metrics_store_keep_individual_batch_metrics true \
--metrics_store_enable_op_level_metrics false \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_generation_stalls_experiments \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_arxiv-summarization_filtered_seq16384_static_num_reqs_128_sarathi_chunk_size_1536_batch_size_256_a6c5dced \
--synthetic_request_generator_length_provider trace \
--trace_request_length_generator_trace_file ./data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv \
--trace_request_length_generator_max_tokens 16384 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 256 \
--sarathi_scheduler_chunk_size 1536 \
--sarathi_scheduler_enable_rolling_prefills true \
--sarathi_scheduler_enable_dynamic_chunking_schedule false

