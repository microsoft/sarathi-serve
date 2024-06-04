# Hybrid Parallelism (Figure 8)

## Experiment Design

In this experiment, we analyze the effectiveness of pipeline parallelism in low-bandwidth conditions. This experiment has two parts, a) where we analyze the TBT latency difference between cross-node tensor parallelism and hybrid parallelism b) identifying the capacity for tensor parallelism and hybrid parallelism settings. Please make sure that you read the readme [`../figure-5-6/README.md`](../figure-5-6/README.md) for the capacity search experiment before proceeding further.

## Experiment Setup

This experiment uses Falcon-180B with 8 A100 80GB PCIe GPUs that are connected with NVLINK in a pairwise manner. Each node contains 4 GPUs on each (Azure NC96ads v4 VMs) and 100 GBps Ethernet connection between them. Before starting the job, start ray on the cluster and onboard all GPUs.

## Running experiments

### Experiment A - Latency and Batch Size

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository.

```sh
bash ./osdi-experiments/figure-8/pipeline_parallel_latency.sh
```

This script executes each batch size and parallelism configuration sequentially, stats of several metrics are printed at the end of each trail. These logs can be used to find the P99 TBT -- look for median of `decode_token_execution_plus_preemption_time` metric in the logs.

### Experiment B - Capacity Search

Please run the following commands from the root of the repository. Note that the Sarathi-Serve and vLLM hybrid parallel results are obtained using the same commands described which we described earlier for [figure-6](../figure-5-6/README.md), we include them here for the sake of completeness. 

```sh
python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 5.0 \
--config-path ./osdi-experiments/figure-8/falcon180b_tp8.yml \
--output-dir ./capacity_search_output_falcon180b_relaxed

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 1.0 \
--config-path ./osdi-experiments/figure-8/falcon180b_tp8.yml \
--output-dir ./capacity_search_output_falcon180b_strict

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 5.0 \
--config-path ./osdi-experiments/figure-5-6/falcon180b_relaxed.yml \
--output-dir ./capacity_search_output_falcon180b_relaxed

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 1.0 \
--config-path ./osdi-experiments/figure-5-6/falcon180b_strict.yml \
--output-dir ./capacity_search_output_falcon180b_strict
```
