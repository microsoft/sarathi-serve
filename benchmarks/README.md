# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Running benchmark_throughput.py
Output will be saved in `./outputs/` folder with timestamp when the script was invoked.
```bash
TOKENIZERS_PARALLELISM=false python benchmarks/benchmark_throughput.py \
--model codellama/CodeLlama-34b-Instruct-hf \
--dataset datasets/prompt_length_1024_pd_ratio_32/prompt_length_1024_pd_ratio_32.json \
--num-prompts 1000 \
--scheduler-type dsarathi \
--max-num-seqs 10 \
--chunk-size 256 \
--enable-rolling-prefills \
--prefill-fitting-tolerance 0.2 \
--write-metrics
```