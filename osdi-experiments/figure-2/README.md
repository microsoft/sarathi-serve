# Per-token Prefill and Decode Time (Figure 2)

## Objective

We show that:

1. per-token prefill time becomes approximately constant from 512 length prefill.
2. per-token decode time decreases drastically as the decode batch size grows from 1 to 128.
3. Show the split of time taken in `linear`, `attention`, `communication` and `others` inline. We have omitted this from the plot to significantly reduce the complexity of the plot while keeping the key insights.

We do this by carefully constructing prefill batches of different lengths, decode batches of different sizes and measuring the batch times.

## Experiment Setup

This experiment uses Yi-34B with 2 A100 80GB PCIe GPUs that are connected with NVLINK in a pairwise manner for 2-way tensor parallelism. It was run in a single node (Azure NC96ads v4 VMs) containing 4 such GPUs. Before starting the job, start ray on the cluster and onboard all GPUs using

```sh
ray stop
ray start --head
```

## Running the experiment

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository. This script has two runs each taking roughly 10 minutes to complete.

```sh
bash ./osdi-experiments/figure-2/operation_time_split.sh
```

This `.sh` file can also be generated using the `generate_runs()` function in the notebook `operation_time_split.ipynb`.

## Interpreting the results

Run the `plot()` function in  `operation_time_split.ipynb` to plot the decode completion time series.
