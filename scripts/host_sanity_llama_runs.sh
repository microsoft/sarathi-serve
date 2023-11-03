#!/bin/bash

set -x

# ray stop && ray start --head && \
# python -m vllm.benchmark.main \
#     --model_name "meta-llama/Llama-2-7b-hf" \
#     --model_tokenizer "meta-llama/Llama-2-7b-hf" \
#     --model_num_layers 32 \
#     --model_tensor_parallel_degree 1 \
#     --metrics_store_wandb_run_name "vllm 7b tp1 `hostname`"

ray stop && ray start --head && \
python -m vllm.benchmark.main \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --model_tokenizer "meta-llama/Llama-2-7b-hf" \
    --model_tensor_parallel_degree 2 \
    --metrics_store_wandb_run_name "vllm 7b tp2 `hostname`"

ray stop && ray start --head && \
python -m vllm.benchmark.main \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --model_tokenizer "meta-llama/Llama-2-7b-hf" \
    --model_tensor_parallel_degree 4 \
    --metrics_store_wandb_run_name "vllm 7b tp4 `hostname`"

# ray stop && ray start --head && \
# python -m vllm.benchmark.main \
#     --model_name "meta-llama/Llama-2-7b-hf" \
#     --model_tokenizer "meta-llama/Llama-2-7b-hf" \
#     --model_tensor_parallel_degree 8 \
#     --metrics_store_wandb_run_name "vllm 7b tp8 `hostname`"
