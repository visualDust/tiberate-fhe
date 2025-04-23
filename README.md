# What is this

This is a variant of [Desilo/liberate-fhe](https://github.com/Desilo/liberate-fhe) with better typing and some user friendly features. This fork has left fork network and will not return to the original branch.

> [!CAUTION]
> This might not be a drop-in replacement for the original one, but it should be compatible with most of the code.
> This is a GPU only library, and it will not work or even compile on CPU only machines.
> This project is still in development, and the API may change in the future. Please use it with caution.

## Pre-requisites

- Python 3.10 or later.
- CUDA 11 or later, CUDA 12+ is recommended. `nvcc` is required for compiling, please make sure it is in your `PATH`.
- [PyTorch](https://pytorch.org/) 2.0 or later is required.

## Install

![PyPI - Version](https://img.shields.io/pypi/v/tiberate) ![PyPI - Downloads](https://img.shields.io/pypi/dw/tiberate)

```bash
conda create -n tiberate python=3.12
conda activate tiberate

pip install tiberate --verbose
```

The install process may take long time depending on your machine. Using verbose flag will help you to see the progress.

> [!NOTE]
> If you encounter the error saying "The detected CUDA version mismatches the version that was used to compile PyTorch.", you can switch cuda version, or manually install pytorch with the same CUDA version as your system, then install this package without dependencies via:
>
> ```bash
> pip install tiberate --no-deps --verbose
> ```

## Dev

For dev or edible installation, you can install from source.

```bash
# clone the repository
git clone https://github.com/visualDust/tiberate-fhe.git
cd tiberate-fhe

# create conda env
conda create -n tiberate python=3.12
conda activate tiberate

# build and install
pip install --editable . --verbose
```

## Usage

```python
import torch
from tiberate import CkksEngine
from tiberate.typing import Plaintext, Ciphertext
from vdtoys.plot import diff_distribution

# Engine creation
engine = CkksEngine()
print(engine)
# Some dummy data
data = torch.randn(8192)
# Encrypt some data
ct = engine.encodecrypt(data)
# Some plaintext with cache
pt = Plaintext(data)
# Save and load ciphertext
ct.save("./ct.pt")
ct = Ciphertext.load("./ct.pt")
# Operations with plaintext
ct = engine.pc_mult(pt, ct)  # Multiplication
ct = engine.pc_add(pt, ct)  # Addition
print(pt)  # Print the plaintext information
# Ciphertext operations
ct = engine.cc_mult(ct, ct)  # Multiplication
ct = engine.cc_add(ct, ct)  # Addition
ct = engine.rotate_single(ct, engine.rotk[1])  # Rotation
ct = engine.rotate_single(ct, engine.rotk[-1])  # Rotate back
# Decryption
whatever = engine.decryptcode(ct, is_real=True)
# Error distribution
data = data * data + data
data *= data
data += data
diff = data - whatever[:8192]
print(f"Mean: {diff.mean()}, Std: {diff.std()}")
plt = diff_distribution(diff)
plt.show()
```

## Run benchmarks

To run benchmarks, you can use the `tiberate benchmark` command:

```bash
tiberate benchmark
```

This will list all available benchmarks and their options. You can run a specific benchmark by providing its file path:

```bash
tiberate benchmark --file path/to/your_benchmark.py
```
