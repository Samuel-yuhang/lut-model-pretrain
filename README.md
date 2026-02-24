# LUTNeuro(🧠) => 🔮

<p align="center">
    <a href="https://github.com/LUT-FPGA/mini-LUT"><img src="https://img.shields.io/badge/miniLUT-Base-blue.svg"></a>
    <a href="https://github.com/LUT-FPGA/mini-LUT/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
    <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
    <a href="#-about">📙About</a> •
    <a href="#-quick-start">🔥Quick Start</a> •
</p>

## 📙 About

**LUTNeuro** is a high-performance neural network library that accelerates linear transformations using **Lookup Tables (LUT)** with **Product Quantization**. By replacing traditional matrix multiplications with fast table lookups, LUTNeuro delivers significant speedups for inference while maintaining model accuracy.

- **🔧 Drop-in Replacement**: Seamlessly replace `nn.Linear` layers in existing PyTorch models
- **📦 Model Integration**: Built-in support for Hugging Face transformers with automatic layer replacement

## 🔥 Quick Start

### Installation

Prerequisites: [uv](https://docs.astral.sh/uv/), [just](https://just.systems/), and [prek](https://prek.j178.dev/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/j178/prek/releases/download/v0.3.3/prek-installer.sh | sh
```

Setup:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e "."
```

### Development

Install pre-commit hooks:

```bash
prek install
just prek # prek run --all-files
```

Python code formatting and linting:

```bash
just fmt # uvx ruff check . --fix && uvx ruff format .
```

### Experiments

See [justfile](justfile) for available experiments.

```bash
just resnet
just lut_resnet
just clm
just lut_clm
```

### Other References

- Kmeans init: https://github.com/LUT-FPGA/SpinQuant/blob/qwen/optimize_centroid.py
- Fused kernel: https://github.com/LUT-FPGA/LUTNeuro/blob/main/lutneuro/ops/lut_linear.py
