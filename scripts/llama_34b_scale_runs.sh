#!/bin/bash

set -x

# for scheudler in "faster_transformer" "orca"; do
#     python -m vllm.benchmark.main \
#         --model_name "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tokenizer "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tensor_parallel_degree 1 \
#         --cluster_num_replicas 32 \
#         --replica_scheduler_provider $scheudler \
#         --replica_scheduler_max_batch_size 3 \
#         --metrics_store_wandb_run_name "vllm 34b tp1 r32 $scheudler"

#     python -m vllm.benchmark.main \
#         --model_name "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tokenizer "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tensor_parallel_degree 2 \
#         --cluster_num_replicas 16 \
#         --replica_scheduler_provider $scheudler \
#         --replica_scheduler_max_batch_size 32 \
#         --metrics_store_wandb_run_name "vllm 34b tp2 r16 $scheudler"

#     python -m vllm.benchmark.main \
#         --model_name "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tokenizer "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tensor_parallel_degree 4 \
#         --cluster_num_replicas 8 \
#         --replica_scheduler_provider $scheudler \
#         --replica_scheduler_max_batch_size 32 \
#         --metrics_store_wandb_run_name "vllm 34b tp4 r8 $scheudler"
# done


# for chunk_size in 256 512 1024 1536 2046; do
#     python -m vllm.benchmark.main \
#         --model_name "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tokenizer "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tensor_parallel_degree 1 \
#         --cluster_num_replicas 32 \
#         --replica_scheduler_provider dsarathi \
#         --sarathi_scheduler_chunk_size $chunk_size \
#         --metrics_store_wandb_run_name "vllm 34b tp1 r32 dsarathi"

#     python -m vllm.benchmark.main \
#         --model_name "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tokenizer "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tensor_parallel_degree 2 \
#         --cluster_num_replicas 16 \
#         --replica_scheduler_provider dsarathi \
#         --sarathi_scheduler_chunk_size $chunk_size \
#         --metrics_store_wandb_run_name "vllm 34b tp2 r16 dsarathi"

#     python -m vllm.benchmark.main \
#         --model_name "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tokenizer "codellama/CodeLlama-34b-Instruct-hf" \
#         --model_tensor_parallel_degree 4 \
#         --cluster_num_replicas 8 \
#         --replica_scheduler_provider dsarathi \
#         --sarathi_scheduler_chunk_size $chunk_size \
#         --metrics_store_wandb_run_name "vllm 34b tp4 r8 dsarathi"
# done


# python -m vllm.benchmark.main \
#     --model_name "meta-llama/Llama-2-70b-hf" \
#     --model_tokenizer "meta-llama/Llama-2-70b-hf" \
#     --model_tensor_parallel_degree 2 \
#     --cluster_num_replicas 16 \
#     --replica_scheduler_max_batch_size 32 \
#     --metrics_store_wandb_run_name "vllm 70b tp2 r16"

# python -m vllm.benchmark.main \
#     --model_name "meta-llama/Llama-2-70b-hf" \
#     --model_tokenizer "meta-llama/Llama-2-70b-hf" \
#     --model_tensor_parallel_degree 4 \
#     --cluster_num_replicas 8 \
#     --replica_scheduler_max_batch_size 64 \
#     --metrics_store_wandb_run_name "vllm 70b tp4 r8"

# python -m vllm.benchmark.main \
#     --model_name "meta-llama/Llama-2-7b-hf" \
#     --model_tokenizer "meta-llama/Llama-2-7b-hf" \
#     --model_tensor_parallel_degree 1 \
#     --cluster_num_replicas 32 \
#     --metrics_store_wandb_run_name "vllm 7b tp1 r32"

# python -m vllm.benchmark.main \
#     --model_name "meta-llama/Llama-2-7b-hf" \
#     --model_tokenizer "meta-llama/Llama-2-7b-hf" \
#     --model_tensor_parallel_degree 2 \
#     --cluster_num_replicas 16 \
#     --metrics_store_wandb_run_name "vllm 7b tp2 r16"

# python -m vllm.benchmark.main \
#     --model_name "meta-llama/Llama-2-7b-hf" \
#     --model_tokenizer "meta-llama/Llama-2-7b-hf" \
#     --model_tensor_parallel_degree 4 \
#     --cluster_num_replicas 8 \
#     --metrics_store_wandb_run_name "vllm 7b tp4 r8"

python -m vllm.benchmark.main \
    --model_name "tiiuae/falcon-40b" \
    --model_tokenizer "tiiuae/falcon-40b" \
    --model_tensor_parallel_degree 1 \
    --cluster_num_replicas 32 \
    --metrics_store_wandb_run_name "vllm f40b tp1 r32"

python -m vllm.benchmark.main \
    --model_name "tiiuae/falcon-40b" \
    --model_tokenizer "tiiuae/falcon-40b" \
    --model_tensor_parallel_degree 2 \
    --cluster_num_replicas 16 \
    --metrics_store_wandb_run_name "vllm f40b tp2 r16"

python -m vllm.benchmark.main \
    --model_name "tiiuae/falcon-40b" \
    --model_tokenizer "tiiuae/falcon-40b" \
    --model_tensor_parallel_degree 4 \
    --cluster_num_replicas 8 \
    --metrics_store_wandb_run_name "vllm f40b tp4 r8"
