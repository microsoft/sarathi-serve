# Capacity Search (Figures 5-6)

## Experiment Design

To identify the maximum capacity for a given configuration (scheduler, model, trace, SLO combination), we run the system with different request rates (QPS) and find the maximum rate where the SLOs are maintained. This is done using binary search -- for each configuration we require 7-8 trails to find the capacity point. We run each trial with 2048 requests and a 30 min time limit. As a result, each configuration requires about 4/5 hrs of execution time. With 3 schedulers, 2 traces and 2 SLO points, each model requires about 60 hrs of execution time. The trails are automatically parallelized across available GPUs for different schedulers and traces within each invocation of the capacity search script. Thus, the mistral (tensor parallel degree 1) experiment would take about 15 hrs with 4 A100 GPUs. Whereas Yi-34B (tensor parallel degree 1) would require roughly twice the amount of time -- 30 hrs.

## Experiment Setup

Mistral-7B, Yi-34B and Falcon-180B experiments are performed using A100 80GB PCIe GPUs that are connected with NVLINK in a pairwise manner, with 4 GPUs on each node (Azure NC96ads v4 VMs). For Mistral and Yi experiments a single node is sufficient, whereas the Falcon experiment requires 2 nodes. In contrast, Llama2-70B experiments are run on a custom node with 8 A40 GPUs pairwise NVLINKED. Before starting the job, start ray on the cluster and onboard all GPUs.

## Running experiments

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository.

```sh
python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 0.5 \
--config-path ./osdi-experiments/figure-5-6/mistral7b_relaxed.yml \
--output-dir ./capacity_search_output_mistral7b_relaxed

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 0.1 \
--config-path ./osdi-experiments/figure-5-6/mistral7b_strict.yml \
--output-dir ./capacity_search_output_mistral7b_strict

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 1.0 \
--config-path ./osdi-experiments/figure-5-6/yi34b_relaxed.yml \
--output-dir ./capacity_search_output_yi34b_relaxed

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 0.2 \
--config-path ./osdi-experiments/figure-5-6/yi34b_strict.yml \
--output-dir ./capacity_search_output_yi34b_strict

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 5.0 \
--config-path ./osdi-experiments/figure-5-6/llama70b_relaxed.yml \
--output-dir ./capacity_search_output_llama70b_relaxed

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 1.0 \
--config-path ./osdi-experiments/figure-5-6/llama70b_strict.yml \
--output-dir ./capacity_search_output_llama70b_strict

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 5.0 \
--config-path ./osdi-experiments/figure-5-6/falcon180b_relaxed.yml \
--output-dir ./capacity_search_output_falcon180b_relaxed

python -m sarathi.benchmark.capacity_search.main \
--tbt-slo-value 1.0 \
--config-path ./osdi-experiments/figure-5-6/falcon180b_strict.yml \
--output-dir ./capacity_search_output_falcon180b_strict
```