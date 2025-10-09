# Sarathi-Serve

Sarathi-Serve is a high througput and low-latency LLM serving framework. Please refer to our [OSDI'24 paper](https://www.usenix.org/conference/osdi24/presentation/agrawal) for more details. 

## Setup

### Setup CUDA

Sarathi-Serve has been tested with CUDA 12.3 on H100 and A100 GPUs.

### Clone repository

```sh
git clone git@github.com:microsoft/sarathi-serve.git
```

### Create mamba environment

Setup mamba if you don't already have it,

```sh
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh # follow the instructions from there
```

Create a Python 3.10 environment,

```sh
mamba create -p ./env python=3.10  
```

### Install Sarathi-Serve

```sh
pip install -e .
```

## Reproducing Results

Refer to readmes in individual folders corresponding to each figure in `osdi-experiments`.

## Citation

If you use our work, please consider citing our paper:

```
@article{agrawal2024taming,
  title={Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve},
  author={Agrawal, Amey and Kedia, Nitin and Panwar, Ashish and Mohan, Jayashree and Kwatra, Nipun and Gulavani, Bhargav S and Tumanov, Alexey and Ramjee, Ramachandran},
  journal={Proceedings of 18th USENIX Symposium on Operating Systems Design and Implementation, 2024, Santa Clara},
  year={2024}
}
```

## Acknowledgment

This repository originally started as a fork of the [vLLM project](https://vllm-project.github.io/). Sarathi-Serve is a research prototype and does not have complete feature parity with open-source vLLM. We have only retained the most critical features and adopted the codebase for faster research iterations.
