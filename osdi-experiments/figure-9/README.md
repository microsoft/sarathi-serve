# Chunk Size Ablations (Figure 9)

## Experiment Design

We have these experiments to analyze the impact of chunking on throughput and latency. First, we analyze the overhead of chunking -- we take 4 prefill lengths (2k, 4k, 8k and 16k) and 3 chunk sizes (512, 1024, 2048) and compare the prefill computation time. Second, we do latency-throughput tradeoff similar to figure 7 at different chunk sizes. Please make sure that you read the readme [`../figure-7/README.md`](../figure-7/README.md) for the isoqps experiment before proceeding further. The first experiment completes in a few minutes, while the later takes about 15-20 hrs.

## Experiment Setup

This experiment uses Mistal-7B and Yi-34B with A100 80GB PCIe GPUs that are connected with NVLINK in a pairwise manner for 2-way tensor parallelism. It was run in a single node (Azure NC96ads v4 VMs) containing 4 such GPUs. Before starting the job, start ray on the cluster and onboard all GPUs using

```sh
ray stop
ray start --head
```

## Running the experiment

### Chunking overhead (Figure 9a)

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository. This script has two runs each taking roughly 10 minutes to complete.

```sh
bash ./osdi-experiments/figure-9/prefill_chunking_overhead_runs.sh
```

This `.sh` file can also be generated using the `generate_runs()` function in the notebook `prefill_chunking_overhead.ipynb`.

#### Interpreting the results

Run the `plot()` function in  `prefill_chunking_overhead.ipynb` to plot prefill processing overhead for different prefill lengths and chunk sizes.

### Latency-Throughput Analysis (Figure 9b)

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository.

```sh
python osdi-experiments/figure-7/isoqps/main.py \
--qps-values 1.0 3.0 \
--config-path ./osdi-experiments/figure-9/mistral7b.yml \
--output-dir ./isoqps_output_chunking_mistral7b

python osdi-experiments/figure-7/isoqps/main.py \
--qps-values 0.55 1.0 \
--config-path ./osdi-experiments/figure-9/yi34b.yml \
--output-dir ./isoqps_output_chunking_yi34b
```
