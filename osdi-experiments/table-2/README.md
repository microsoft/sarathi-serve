# Per-token Prefill and Decode Time (Figure 2)

## Experiment design

We show that piggybacking significantly improves decode performance compared to executing decode-only batches.

1. Our first specimen batch is a prefill-only batch of 1024 tokens.
2. Our 2nd specimen batch is a decode-only batch of 4 tokens.
3. Our 3rd specimen batch is a hybrid batch of 1021 prefill tokens followed by 3 decode tokens.

We use vLLM scheduler to produce prefill-only and decode-only batches. We use the sarathi scheduler to create
a hybrid batch. The number of requests, batch_size, prefill and decode lengths are meticulously set to yield specimen batches.

We show that per-token time remains the same for prefill-only and hybrid batches, even though the hybrid batch has a small number of decode tokens mixed in it. The piggybacked decode tokens are being executed as fast as prefill tokens.
Contrast this with the decode-only batch where the per-token decode time is 85x higher than these piggybacked decode tokens.

## Experiment Setup

This experiment uses Yi-34B with 2 A100 80GB PCIe GPUs that are connected with NVLINK in a pairwise manner for 2-way tensor parallelism. It was run in a single node (Azure NC96ads v4 VMs) containing 4 such GPUs. Before starting the job, start ray on the cluster and onboard all GPUs using

```sh
ray stop
ray start --head
```

## Running the experiment

After setting up the repository on the appropriate hardware platform as described above, run the following commands from the root of the repository. This script has two runs each taking roughly 10 minutes to complete.

```sh
bash ./osdi-experiments/table-2/runs.sh
```

This `.sh` file can also be generated using the `generate_runs()` function in the notebook `decode_maximal_vs_naive_batching.ipynb`.

## Interpreting the results

Run the `get_df()` function in  `decode_maximal_vs_naive_batching.ipynb` to get the results. Sample results in `df.csv` are provided for reference.
