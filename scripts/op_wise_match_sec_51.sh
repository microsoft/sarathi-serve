#!/bin/bash

set -x

ray stop && ray start --head && \
python -m vllm.benchmark.main \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --model_tokenizer "meta-llama/Llama-2-7b-hf" \
    --model_num_layers 32 \
    --model_tensor_parallel_degree 1 \
    --metrics_store_wandb_run_name "vllm 7b tp1 new"

ray stop && ray start --head && \
python -m vllm.benchmark.main \
    --model_name "meta-llama/Llama-2-70b-hf" \
    --model_tokenizer "meta-llama/Llama-2-70b-hf" \
    --model_num_layers 80 \
    --model_tensor_parallel_degree 4 \
    --metrics_store_wandb_run_name "vllm 70b tp4 new"

ray stop && ray start --head && \
python -m vllm.benchmark.main \
    --model_name "codellama/CodeLlama-34b-Instruct-hf" \
    --model_tokenizer "codellama/CodeLlama-34b-Instruct-hf" \
    --model_num_layers 48 \
    --model_tensor_parallel_degree 2 \
    --metrics_store_wandb_run_name "vllm 34b tp2 new"
