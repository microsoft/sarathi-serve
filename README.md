# Sarathi-Serve

This repository contains the code for [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310).
This codebase also serves as the baseline for fidelity tests for the LLM inference system simulator [Vidur](https://github.com/microsoft/vidur).

## Setup

### Setup CUDA

Sarathi-Serve has been tested with CUDA 12.1 on A100 and A40 GPUs.

### Clone repository

```sh
git clone https://github.com/microsoft/sarathi.git
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

### Install Dev Dependencies

```sh
pip install -r requirements-dev.txt
```

### Install PyTorch

```sh
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

### Update NCCL

NCCL 2.19 which ships with torch 2.2 by default is buggy, update NCCL to the latest (2.21),

```sh
pip install -U nvidia-nccl-cu12
```

### Install FlashAttention

```sh
pip install ninja packaging
git submodule update --init --recursive
cd third_party/flash-attention
python setup.py install
```

### Install Sarathi-Serve

```sh
pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.2/
```

## Citation

If you use our work, please consider citing our paper:

```latex
@article{agrawal2024taming,
  title={Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve},
  author={Agrawal, Amey and Kedia, Nitin and Panwar, Ashish and Mohan, Jayashree and Kwatra, Nipun and Gulavani, Bhargav S and Tumanov, Alexey and Ramjee, Ramachandran},
  journal={Proceedings of 18th USENIX Symposium on Operating Systems Design and Implementation, 2024, Santa Clara},
  year={2024}
}
```

## Acknowledgment

This repository originally started as a fork of the [vLLM project](https://vllm-project.github.io/). Sarathi-Serve is a research prototype and does not have complete feature parity with open-source vLLM. We have only retained the most critical features and adopted the codebase for faster research iterations.

## Contributing

Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
