#!/bin/bash

set -x

for qps in 0.5 1 1.5 2 2.5; do
    ray stop && ray start --head && \
    python -m vllm.benchmark.main \
        --model_name "meta-llama/Llama-2-7b-hf" \
        --model_tokenizer "meta-llama/Llama-2-7b-hf" \
        --model_num_layers 32 \
        --model_tensor_parallel_degree 1 \
        --metrics_store_wandb_run_name "vllm 7b tp1 qps:$qps" \
        --poisson_request_interval_generator_qps $qps
done
