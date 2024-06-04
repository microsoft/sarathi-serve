#!/bin/bash
set -x
CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16385 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider trace \
--synthetic_request_generator_interval_provider static \
--trace_request_length_generator_trace_file /home/amey/sarathi-lean/osdi-experiments/figure-2/prefill_operation_time_split_experiment_trace.csv \
--trace_request_length_generator_max_tokens 16385 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--synthetic_request_generator_num_requests 8 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 1 \
--vllm_scheduler_max_tokens_in_batch 16385 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_prefill_split_35fdcdac \
--metrics_store_enable_op_level_metrics false 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 16385 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider trace \
--synthetic_request_generator_interval_provider static \
--trace_request_length_generator_trace_file /home/amey/sarathi-lean/osdi-experiments/figure-2/prefill_operation_time_split_experiment_trace.csv \
--trace_request_length_generator_max_tokens 16385 \
--trace_request_length_generator_prefill_scale_factor 1 \
--trace_request_length_generator_decode_scale_factor 1 \
--synthetic_request_generator_num_requests 8 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 1 \
--vllm_scheduler_max_tokens_in_batch 16385 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_prefill_split_35fdcdac \
--metrics_store_enable_op_level_metrics true 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 2 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 1 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_1_decode_split_f2142876 \
--metrics_store_enable_op_level_metrics false 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 2 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 1 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_1_decode_split_f2142876 \
--metrics_store_enable_op_level_metrics true 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 4 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 2 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_2_decode_split_cd69e1d1 \
--metrics_store_enable_op_level_metrics false 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 4 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 2 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_2_decode_split_cd69e1d1 \
--metrics_store_enable_op_level_metrics true 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 8 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 4 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_4_decode_split_81daea07 \
--metrics_store_enable_op_level_metrics false 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 8 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 4 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_4_decode_split_81daea07 \
--metrics_store_enable_op_level_metrics true 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 16 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 8 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_8_decode_split_015f87c4 \
--metrics_store_enable_op_level_metrics false 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 16 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 8 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_8_decode_split_015f87c4 \
--metrics_store_enable_op_level_metrics true 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 32 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 16 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_16_decode_split_ad8b7534 \
--metrics_store_enable_op_level_metrics false 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 32 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 16 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_16_decode_split_ad8b7534 \
--metrics_store_enable_op_level_metrics true 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 64 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 32 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_32_decode_split_d0049dc7 \
--metrics_store_enable_op_level_metrics false 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 64 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 32 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_32_decode_split_d0049dc7 \
--metrics_store_enable_op_level_metrics true 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 128 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 64 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_64_decode_split_96a02bf7 \
--metrics_store_enable_op_level_metrics false 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 128 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 64 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_64_decode_split_96a02bf7 \
--metrics_store_enable_op_level_metrics true 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 256 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 128 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_128_decode_split_7e79ce63 \
--metrics_store_enable_op_level_metrics false 

CUDA_VISIBLE_DEVICES=0,1 python sarathi/benchmark/main.py \
--output_dir /home/amey/sarathi-lean/osdi-experiments/figure-2/benchmark_output \
--model_name 01-ai/Yi-34B-200K \
--model_max_model_len 1024 \
--cluster_num_replicas 1 \
--model_tensor_parallel_degree 2 \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_length_provider synthetic \
--synthetic_request_generator_length_provider uniform \
--synthetic_request_generator_interval_provider static \
--uniform_request_length_generator_prefill_to_decode_ratio 255.0 \
--uniform_request_length_generator_min_tokens 1024 \
--uniform_request_length_generator_max_tokens 1024 \
--synthetic_request_generator_num_requests 256 \
--metrics_store_keep_individual_batch_metrics true \
--replica_scheduler_provider vllm \
--replica_scheduler_max_batch_size 128 \
--vllm_scheduler_max_tokens_in_batch 1024 \
--metrics_store_wandb_project llm-simulator \
--metrics_store_wandb_group vllm_operation_time_split_experiment \
--metrics_store_wandb_run_name Yi-34B-200K_tp_2_seq_len_1024_decode_batch_size_128_decode_split_7e79ce63 \
--metrics_store_enable_op_level_metrics true 
