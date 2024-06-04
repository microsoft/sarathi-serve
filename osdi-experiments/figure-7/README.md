# Lateny-Throughput Tradeoff Analysis (Figures 7)

## Experiment Design

This experiment is designed to understand how different scheduler respond to varying QPS values, specially in terms of impact on latency. We run each model - scheduler pair at different QPS values and measure the TBT and TTFT values. We further evaluate 4 different batch sizes for vLLM to get full range of latency-throughput tradeoff exposed by the batch size knob. We run each trial with 1024 requests and a 3 hr time limit. With 5 schedulers (4 batch sizes for vLLM + 1 for Sarathi-Serve), and 4 QPS values, each model requires about 60 hrs of execution time. The trails are automatically parallelized across available GPUs for different schedulers and traces within each invocation of the capacity search script. Thus, the mistral (tensor parallel degree 1) experiment would take about 15 hrs with 4 A100 GPUs. Whereas Yi-34B (tensor parallel degree 1) would require roughly twice the amount of time -- 30 hrs.

## Experiment Setup

Mistral-7B, Yi-34B and Falcon-180B experiments are performed using A100 80GB PCIe GPUs that are connected with NVLINK in a pairwise manner, with 4 GPUs on each node (Azure NC96ads v4 VMs). For Mistral and Yi experiments a single node is sufficient, whereas the Falcon experiment requires 2 nodes. In contrast, Llama2-70B experiments are run on a custom node with 8 A40 GPUs pairwise NVLINKED. Before starting the job, start ray on the cluster and onboard all GPUs.

## Running experiments

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository.

```sh
python osdi-experiments/figure-7/isoqps/main.py \
--qps-values 1.0 2.0 3.0 4.0 \
--config-path ./osdi-experiments/figure-7/mistral7b.yml \
--output-dir ./isoqps_output_mistral7b

python osdi-experiments/figure-7/isoqps/main.py \
--qps-values 0.55 0.7 0.85 1.0 \
--config-path ./osdi-experiments/figure-7/yi34b.yml \
--output-dir ./isoqps_output_yi34b

python osdi-experiments/figure-7/isoqps/main.py \
--qps-values 0.2 0.4 0.6 0.8 \
--config-path ./osdi-experiments/figure-7/llama70b.yml \
--output-dir ./isoqps_output_llama70b

python osdi-experiments/figure-7/isoqps/main.py \
--qps-values 0.4 0.6 0.8 1.0 \
--config-path ./osdi-experiments/figure-7/falcon180b.yml \
--output-dir ./isoqps_output_falcon180b
```
