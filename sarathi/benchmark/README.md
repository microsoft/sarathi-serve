# Understanding the parameters taken by the simulator

The [default.yml](sarathi/benchmark/config/default.yml) is the comprehensive list of all parameters taken by the benchmark suite. While invoking it, any of these parameters can be overridden. Running only `python -m sarathi.main` means that all the parameters are taken from the `default.yml` file and have no overrides.
The parameters descriptions are given below:

1. `seed`: Random seed which is set in multiple random generators notably the request length and inter-request time generators. This is useful for reproducibility.
1. `log_level`: Logging level. Not comprehensively supported currently.
1. `output_dir`: The directory where each invocation of sarathi creates its run directory. Eg: `<project_root>/benchmark_output/2023-11-20_11-31-40-523377`.
All the output files corresponding to the invocation are stored under this directory eg. the chrome trace, cdf plots etc. We call each invocation of sarathi a run.
1. `write_json_trace`: Whether to write the requests sent to the system in a JSON file.
1. `write_chrome_trace`: Whether to write the chrome trace. This is useful for debugging. Use `chrome://tracing` or `edge://tracing` to view the trace.
1. `write_metrics`: This is a blanket flag to enable/disable the writing of all metrics.
1. `gpu_memory_utilization`: Fraction of memory that is used for KV cache, weights and activation memory. The remaining is left for `nccl`, `cuBLAS` etc. libraries. This is not a strict constraint. Actual deployment does go over this limit.
1. `time_limit`: The time in seconds for which the benchmark is to be run. This is useful to run the benchmark for a fixed amount of time. The benchmark will stop after this time limit is reached.
1. `replica_resource_mapping`: Ignore this parameter for now.
1. `cluster`:
    1. `num_replicas`: Number of replicas in the clusters. Replicas and independent and identical.
    Suppose you have a DGX box with 8 GPUS and you want to serve `meta-llama/Llama-2-70b-hf`.
    One deployment strategy is to run 2 replicas each with 4 GPUs running the model in tensor parallel degree 4.
    Another deployment strategy is to run 1 replica with all 8 GPUs running the model in tensor parallel degree 8.
1. `model`:
    1. `name`: Typically hugging-face id of the model. Eg: `meta-llama/Llama-2-70b-hf`.
    1. `pipeline_parallel_degree`: Pipeline parallel degree. This number must divide the number of layers in the model.
    1. `tensor_parallel_degree`: Tensor parallel degree.
    1. `max_model_len`: Maximum number of tokens a request can have (includes both prefill and decode tokens). Longer requests are ignored.
    1. `load_format`: Use `dummy` for random weights.
    1. `attention_backend`: Specify the kernel to be used for attention. Currently support `flash_attention`, `flashinfer` and `noop`.
1. `request_generator`: The benchmark suite contains a comprehensive request generator. See [here](sarathi/benchmark/request_generator)
    1. `provider`: The request generator to use. Currently supported are `synthetic` and `trace`. `synthetic` generates requests from a synthetic distribution. `trace` generates requests from a real-world trace.
1. `synthetic_request_generator`: This section is used to further define the synthetic request generator. Only required if `request_generator_provider` is set to `synthetic`.
    1. `length_provider`: The distribution of the request length. Currently supported are `uniform`, `trace` and `zipf`.
    1. `interval_provider`: The distribution of the inter-request time. Currently supported are `static`, `trace`, `poisson` and `gamma`.
    1. `num_requests`: Number of requests to generate or select from the trace.
1. `trace_request_generator`: This section is used to further define the trace request generator. Only required if `request_generator_provider` is set to `trace`.
    1. `trace_file`: Path to the trace file.
    2. `date`: Requests with the given date.
    3. `prefill_scale_factor`: Scale factor to apply to the prefill tokens in the trace. Recommend leaving this value at 1.
    4. `decode_scale_factor`: Scale factor to apply to the decode tokens in the trace. Recommend leaving this value at 1.
    5. `time_scale_factor`: Scale factor to apply to the window time in the trace. This can be used to speed up / slow down the trace. Example, to compress a 24h trace to 1h. Scale factors drastically change the workload. Scaled traces cannot be directly compared to the original trace.
1. `trace_request_length_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_length_provider` is set to `trace`.
    1. `trace_file`: Path to the trace file. This trace file is a csv like [cnn_dailymail_stats_llama2_tokenizer.csv](data/processed_traces/cnn_dailymail_stats_llama2_tokenizer.csv)
    1. `prefill_scale_factor`: See `trace_request_generator` section above.
    1. `decode_scale_factor`: See `trace_request_generator` section above.
    1. `max_tokens`: Maximum number of tokens in a request. Requests generated from the trace are clipped at this number. `P:D ratio` is preserved in case of clipping.
1. `zipf_request_length_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_length_provider` is set to `zipf`.
    1. `theta`: Shape parameter of the zipf distribution.
    2. `scramble`: Whether to scramble the zipf distribution. This is useful to avoid the zipf distribution being skewed towards the start of the vocabulary.
1. `trace_request_interval_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_interval_provider` is set to `trace`.
    1. `trace_file`: Path to the trace file.
    1. `start_time`: Start time of the trace to use.
    1. `end_time`: End time of the trace to use.
    1. `time_scale_factor`: See `trace_request_generator` section above.
1. `poisson_request_interval_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_length_provider` is set to `poisson`.
    1. `qps`: Requests per second to hit the system with.
1. `gamma_request_interval_generator`: Only required if `request_generator_provider` is set to `synthetic` and `synthetic_request_length_provider` is set to `gamma`.
    1. `cv`: Coefficient of variation of the gamma distribution.
    1. `qps`: Requests per second to hit the system with.
1. `replica_scheduler`: This is the scheduler which determines how to schedule the requests on a replica.
    1. `provider`: `orca`, `sarathi`, and `vllm`. See [here](simulator/schedulers/replica_schedulers) for more details.
    1. `max_batch_size`: Maximum permissible batch size. Set carefully for `orca`, `fastertransformer`. Have a high limit for schedulers which use dynamic KV cache allocation. They will auto-adjust.
    1. `num_blocks`: TODO. Ignore this parameter for now.
1. `sarathi_scheduler`: <https://arxiv.org/abs/2308.16369>. Only required if `replica_scheduler_provider` is set to `sarathi`.
    1. `chunk_size`: The maximum number of tokens (prefill / decode) to process in a batch. Prefills are done progressively if the number of prefills tokens in a request is greater than this number.
    1. `enable_dynamic_chunking_schedule`: Recommend to set this to false. This is an experimental feature.
1. `vllm_scheduler`: <https://github.com/vllm-project/vllm>. Only required if `replica_scheduler_provider` is set to `vllm`.
    1. `max_tokens_in_batch`: Maximum number of tokens in a batch. This is an additional limit on top of `max_batch_size`.
1. `metrics_store`: Configuration of the metrics store. The metrics store is a central store that stores the metrics of the simulator. At the benchmark end, it dumps the metrics to various files typically `csv`, `png` and `json`. The metrics store is also responsible for uploading the metrics to `wandb`.
    1. `wandb_project`: Wandb project to upload to eg. `llm-simulator`
    1. `wandb_group`
    1. `wandb_run_name`: Pass an empty string to auto-generate the run name. Recommend to have a run name to identify the run.
    1. `enable_op_level_metrics`: Whether to enable operation-level metrics to capture the time taken by each operation like `mlp_up_proj`, `attn_prefill` etc. in the model. See `OperationMetrics` in [constants.py](sarathi/metrics/constants.py).
    1. `enable_request_outputs`: can be used to log the output tokens, text of each request.
    1. `keep_individual_batch_metrics`: Whether to keep individual batch metrics. With this option enabled one can find out for every batch, its composition, execution time, time taken by each operation etc. Otherwise, say the batch_execution_times are aggregated into a cdf.
