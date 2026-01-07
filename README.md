# Sarathi-Serve: Deadline-aware Scheduling

This is a specialized branch of Sarathi-Serve that implements deadline-aware scheduling for LLM inference.

### Environment Setup

Create a Python 3.10 environment:

```sh
# Create the environment named 'niyama'
conda create -n niyama python=3.10 -y

# Activate the environment
conda activate niyama
```

### Install Sarathi-Serve

```sh
pip install -e . https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.1/flashinfer-0.1.1+cu121torch2.3-cp310-cp310-linux_x86_64.whl
```

### Model Access & Environment Variables

To run the benchmarks, you must have access to the models (e.g., Llama-3-8B, Qwen-2.5) on Hugging Face. You must also export your Hugging Face token:

```sh
export HF_TOKEN="your_huggingface_token_here"
```

### Dataset Preparation

Before running benchmarks, download the required Azure trace datasets into the `data/processed_traces` directory:

```sh
mkdir -p data/processed_traces

# Download Code traces
wget -O data/processed_traces/AzureLLMInferenceTrace_code_1week.csv https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv

# Download Conversation traces
wget -O data/processed_traces/AzureLLMInferenceTrace_conv_1week.csv https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv
```

### Reproducibility

To ensure accurate measurement of CUDA kernel elapsed times and guarantee reproducibility of numbers, please follow these configuration steps before running benchmarks:

1.  **Reset GPU Configurations**
    Reset all GPUs to their default configurations to clear any previous states.
    ```sh
    nvidia-smi --gpu-reset
    ```

2.  **Enable Persistence Mode**
    Enable persistence mode to keep the target GPU initialized even when no clients are connected.
    ```sh
    # Replace <gpu_id> with your specific GPU ID (e.g., 3)
    nvidia-smi -pm ENABLED -i <gpu_id>
    ```

3.  **Lock GPU Clocks**
    Lock the GPU clocks to their Thermal Design Power (TDP) frequency to ensure uniform numbers across runs.
    ```sh
    # Lock GPU clocks to TDP
    nvidia-smi --lock-gpu-clocks=tdp,tdp
    
    # Alternatively, set the application clock to the base TDP frequency (e.g., 1065 MHz on A100)
    nvidia-smi ac 1512,1065 -i <gpu_id>
    nvidia-smi -lgc tdp,tdp -i <gpu_id>
    ```

4.  **Code-Level Determinism**
    * **Set Random Seeds**: Empirically, numbers may vary across runs. To mitigate this, set the seed value for both standard and CUDA random number generators:
        ```python
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        ```
    * **Kernel Isolation**: Avoid running different kernels for the sake of measurement inside a single process, as this causes variance in numbers.

### Running Experiments

* Run `tester.sh` to do a test run.
* Run `fig10_11.sh` to recreate Figure 10 and 11 from the paper.
* Run `fig10_11_tiny.sh {gpu_ids_to_use}` to recreate Figure 10 and 11 from the paper with a smaller qps sweep and less number of requests.
* Run `fig7.sh` to recreate Figure 7 from the paper (only Llama3 8b TP1, AzCode, AzConv, ShareGPT, all traces, qps sweep and num requests hardcoded in the sh file).
* Run `fig7_tiny.sh {gpu_ids_to_use}` to run capacity points for different schedulers(only Llama3 8b TP1, AzCode, AzConv, ShareGPT, all traces, qps sweep and num requests hardcoded in the sh file) 
* Run `fig7_{code, conv, sharegpt}.sh` to recreate Figure 7 from the paper (only Llama3 8b TP1, AzCode, AzConv, ShareGPT, single trace, qps sweep and num requests hardcoded in the sh file).
* Run `fig8.sh` to recreate Figure 8 from the paper (only Llama3 8b TP1, AzConv).

## Deadline Scheduling

This branch of Sarathi-Serve implements deadline-based scheduling approach. The branch leverages chunking concepts (for more information about chunking, see [Sarathi paper](https://arxiv.org/abs/2308.16369)). Key features include:

1. **SLA-aware Scheduler**: Maximizes deadline-slack and improves serving throughput by making scheduling decisions based on SLA constraints.

2. **Key Features**:
   - Deadline-aware ordering of requests
   - Strategic request selection under high loads to maintain throughput
   - Dynamic prefill chunk size selection to balance throughput and latency guarantees
   - Tunable parameters to adjust priority based on request length and arrival time

3. **Implementation Details**:
   ### Scheduling Logic
   - The main scheduling loop is implemented in **`sarathi/core/scheduler/deadline_scheduler.py`**:
   - **Lines 271–277**: Invoke the **chunk size predictor** to determine how many tokens to schedule.
   - **Lines 54–167**: Contain the implementation of the chunk size predictor itself.
   - **Line 300**: Performs **eager relegation**, where requests that are predicted to miss their deadline are deprioritized.

   ### Request Metadata and Prioritization
   Defined in **`sarathi/core/datatypes/sequence.py`**:
   - **Line 11**: Defines `TIER_DEADLINES`, the SLA tiers that requests may fall into.
   - **Lines 70–75**: Randomly assign each request to an SLA tier.
   - **Line 79**: Defines the **hybrid prioritization constant**.
   - **Lines 88–109**: Contain the **sequence comparator**, which orders requests based on the hybrid prioritization mechanism.
   - **Lines 106–107** (not commented out anymore): Control the weight between arrival time and remaining tokens (the latter acting as a proxy for remaining processing time).
   - **TTLT (Time-To-Last-Token) SLA**:  
   - Output length is predicted using `num_dec_tokens`.  
   - Approximate **Time-To-First-Token (TTFT)** is estimated by subtracting `num_dec_tokens × TBT_SLA` seconds from the SLA (assuming each decode token takes TBT_SLA seconds).

   ### Chunk Size Prediction
   - Implemented in **`chunk_size_predictor/predictor.py`**.
   - This module is responsible for **fitting the linear regression model** on data collected from previous runs.
   - The trained model is written inside the **deadline scheduler**, which uses it to predict processing times and decide chunk sizes when scheduling requests.

## Acknowledgment

This repository originally started as a fork of the [vLLM project](https://vllm-project.github.io/). Sarathi-Serve is a research prototype and does not have complete feature parity with open-source vLLM. We have only retained the most critical features and adapted the codebase for faster research iterations.
"""