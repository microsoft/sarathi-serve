# Scheduler Ablation (Table 6)

## Experiment Design

We have designed this experiment to understand the impact of hybrid batching and chunking in isolation, and understand how each of them affect TTFT and TBT.


## Experiment Setup

Both experiments use Yi-34B with 2 A100 80GB PCIe GPUs that are connected with NVLINK in a pairwise manner for 2-way tensor parallelism. It was run in a single node (Azure NC96ads v4 VMs) containing 4 such GPUs. Before starting the job, start ray on the cluster and onboard all GPUs using,

```sh
ray stop
ray start --head
```

## Running the experiment

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository. This script has two runs each taking roughly 10 minutes to complete.

```sh
bash ./osdi-experiments/table-6/scheduling_ablation.sh
```

This script executes each model and scheduler configuration sequentially, stats of several metrics are printed at the end of each trail. These logs can be used to find the P99 TBT (`decode_token_execution_plus_preemption_time`) and P50 TTFT (`prefill_e2e_time`) the logs.

