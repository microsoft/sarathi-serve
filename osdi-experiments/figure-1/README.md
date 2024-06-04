# vLLM Tail Latency (Figure 1)

## Experiment Setup

Both experiments use Yi-34B with 2 A100 80GB PCIe GPUs that are connected with NVLINK in a pairwise manner for 2-way tensor parallelism. It was run in a single node (Azure NC96ads v4 VMs) containing 4 such GPUs. Before starting the job, start ray on the cluster and onboard all GPUs using

```sh
ray stop
ray start --head
```

## Generation Stalls (Figure 1a)

### Experiment Design

We show that the `vLLM` scheduler has considerable gaps where no decode tokens are output, while the `sarathi` scheduler has no such gaps and continues to produce decode tokens unfettered. We show this by running the `vLLM` and `sarathi` schedulers on the same model (Yi-34B TP2) and trace (`arxiv summarization`) and comparing the decode completion times.

### Running the experiment

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository. This script has two runs each taking roughly 10 minutes to complete.

```sh
bash ./osdi-experiments/figure-1/generation_stalls.sh
```

This `.sh` file can also be generated using the `generate_runs()` function in the notebook `generation_stalls.ipynb`.

### Interpreting the results

Run the `plot()` function in  `generation_stalls.ipynb` to plot the decode completion time series. The `vllm` series will have steps while `sarathi` one will be smooth. You can also analyze the raw CSVs in path `osdi-experiments/figure-1/benchmark_output/<timestamp>/replica_0/plots/decode_completion_time_series.csv`.

## High tail latency (Figure 1b)

### Experiment Design

We show that when hit with the same workload, the `vLLM` scheduler has a higher tail latency compared to the `sarathi` scheduler. We show this by running the `vLLM` and `sarathi` schedulers on the same model (Yi-34B TP2) and trace (`arxiv summarization`) and comparing the tail latencies for three QPS values.

### Running the experiment

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository. This script has 6 runs and will take roughly 30 minutes to complete.

```sh
bash ./osdi-experiments/figure-1/high_tail_latency.sh
```

This `.sh` file can also be generated using the `generate_runs()` function in the notebook `high_tail_latency.ipynb`.

### Interpreting the results

Run the `plot()` function in  `high_tail_latency.ipynb` to plot the decode completion time series. The `vllm` series will have steps while `sarathi` one will be smooth. You can also analyze the raw CSVs in path `high_tail_latency_output/<timestamp>/replica_0/plots/decode_token_execution_plus_preemption_time.csv`.
