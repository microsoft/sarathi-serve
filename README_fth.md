# Sarathi-Serve

Sarathi-Serve is a high througput and low-latency LLM serving framework. Please refer to our [OSDI'24 paper](https://www.usenix.org/conference/osdi24/presentation/agrawal) for more details. 
Sarathi-Serve 是一个高吞吐量和低延迟的大型语言模型（LLM）服务框架。更多细节，请参考我们的 [OSDI'24 论文](https://www.usenix.org/conference/osdi24/presentation/agrawal)。

## Setup  设置

### Setup CUDA  设置 CUDA

Sarathi-Serve has been tested with CUDA 12.3 on H100 and A100 GPUs.
Sarathi-Serve 已在 H100 和 A100 GPU 上使用 CUDA 12.3 进行了测试。

### Clone repository 克隆仓库

```sh
git clone git@github.com:microsoft/sarathi-serve.git

fth
git@github.com:tianhao909/sarathi-serve.git
```

### Create mamba environment  创建 mamba 环境

Setup mamba if you don't already have it,   如果你还没有安装 mamba，请先安装：

```sh
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh # follow the instructions from there  按照提示进行操作  
```

Create a Python 3.10 environment,  创建一个 Python 3.10 环境：

```sh
mamba create -p ./env python=3.10  

fth 
conda create -p ./env python=3.10 
sudo conda create -p ./env python=3.10  

conda activate /app/software1/sarathi-serve/env
conda activate /disk1/futianhao/software1/test_sarathi-serve/sarathi-serve/env

conda activate /app/software1/sarathi-serve_viudr/sarathi-serve/env
```


### Install Sarathi-Serve 安装 Sarathi-Serve

```sh
pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/

pip install /disk1/futianhao/software1/test_sarathi-serve/sarathi-serve/flashinfer-0.1.6+cu121torch2.3-cp310-cp310-linux_x86_64.whl
/app/software1/test_sarathi-serve/sarathi-serve

pip install /app/software1/test_sarathi-serve/sarathi-serve/flashinfer-0.1.6+cu121torch2.3-cp310-cp310-linux_x86_64.whl
pip install -e . /disk1/futianhao/software1/test_sarathi-serve/sarathi-serve/flashinfer-0.1.6+cu121torch2.3-cp310-cp310-linux_x86_64.whl
fth

curl -L -o flashinfer-0.0.4+cu121torch2.3-cp310-cp310-linux_x86_64.whl https://github.com/flashinfer-ai/flashinfer/releases/download/v0.0.4/flashinfer-0.0.4+cu121torch2.3-cp310-cp310-linux_x86_64.whl#sha256=bd5d9c61675e4eed586e645fd52f1b40878e608a90218cf2db21ff931930ff45

curl -L -o flashinfer-0.1.6+cu121torch2.3-cp310-cp310-linux_x86_64.whl https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.3-cp310-cp310-linux_x86_64.whl#sha256=8bbeb776d315af8213ddc66c84e6ff00d0956839cb753d5b187ece58e3b698c1

https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.3-cp310-cp310-linux_x86_64.whl#sha256=8bbeb776d315af8213ddc66c84e6ff00d0956839cb753d5b187ece58e3b698c1
```

## Reproducing Results   重现结果

Refer to readmes in individual folders corresponding to each figure in `osdi-experiments`.
参考 `osdi-experiments` 中每个文件夹对应的 readme 文件，以了解如何重现每个图表的结果。

## Citation 引用

If you use our work, please consider citing our paper:  如果你使用了我们的工作，请考虑引用我们的论文：

```
@article{agrawal2024taming,
  title={Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve},
  author={Agrawal, Amey and Kedia, Nitin and Panwar, Ashish and Mohan, Jayashree and Kwatra, Nipun and Gulavani, Bhargav S and Tumanov, Alexey and Ramjee, Ramachandran},
  journal={Proceedings of 18th USENIX Symposium on Operating Systems Design and Implementation, 2024, Santa Clara},
  year={2024}
}
```

## Acknowledgment  致谢

This repository originally started as a fork of the [vLLM project](https://vllm-project.github.io/). Sarathi-Serve is a research prototype and does not have complete feature parity with open-source vLLM. We have only retained the most critical features and adopted the codebase for faster research iterations.
这个仓库最初是基于 [vLLM 项目](https://vllm-project.github.io/) 分支出来的。Sarathi-Serve 是一个研究原型，并不完全具备开源 vLLM 的功能。我们只保留了最关键的特性，并针对更快的研究迭代对代码库进行了调整。

