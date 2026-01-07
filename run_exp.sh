python -m sarathi.benchmark.main \
    --output_dir benchmark_output/paper \
    --scheduler_config_type 'SARATHI' \
    --interval_generator_config_type 'POISSON' \
    --poisson_request_interval_generator_config_qps 1.0 \
    --synthetic_request_generator_config_num_requests 150000 \
    --worker_config_attention_backend 'FLASHINFER' \
    --fixed_request_length_generator_config_decode_tokens 375 \
    --fixed_request_length_generator_config_prefill_tokens 3750 \
    --vllm_scheduler_config_max_num_seqs 1000 \
    --vllm_scheduler_config_max_batched_tokens 8192 \
    --time_limit 1000000000000 \
    --model_config_load_format 'DUMMY' \
    --length_generator_config_type 'TRACE' \
    --request_generator_config_type 'SYNTHETIC' \
    --model_config_model 'meta-llama/Meta-Llama-3-8B' \
    --parallel_config_pipeline_parallel_size 1 \
    --parallel_config_tensor_parallel_size 1 \
    --worker_config_gpu_memory_utilization 0.97 \
    --metrics_config_keep_individual_batch_metrics \
    --metrics_config_enable_cpu_op_level_metrics \
    --deadline_scheduler_config_chunk_size 256\
    --sarathi_scheduler_config_chunk_size 256 \
    --deadline_scheduler_config_execution_threshold 0.05\
    --model_config_max_model_len 260000 \
    --trace_request_length_generator_config_trace_file data/processed_traces/AzureLLMInferenceTrace_conv_1week.csv\
    # --replica_resource_mapping "[("10.0.0.69", 1)]"
    # --trace_request_length_generator_config_trace_file data/newtrace/subset_app_2.csv