# What is this

This is a variant of [Desilo/liberate-fhe](https://github.com/Desilo/liberate-fhe) with better typing and some user friendly features. This fork has left fork network and will not return to the original branch.

> [!CAUTION]
> This might not be a drop-in replacement for the original one, but it should be compatible with most of the code.
> This is a GPU only library, and it will not work or even compile on CPU only machines.
> This project is still in development, and the API may change in the future. Please use it with caution.

## Pre-requisites

- Python 3.10 or later.
- CUDA 11 or later, CUDA 12+ is recommended. `nvcc` is required for compiling.
- [PyTorch](https://pytorch.org/) 2.0 or later installed.
- Optionally, [Poetry](https://python-poetry.org/) for development.

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

## Install from source

For dev or edible installation, you can install from source.

```bash
# clone the repository
git clone https://github.com/wens-lab/tiberate-fhe.git
cd tiberate-fhe

# create conda env
conda create -n tiberate python=3.12
conda activate tiberate

# install poetry
pip install poetry

# install package
poetry install # this will also install the package itself into current environment
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

Output:

```
2025-03-18 11:33:35.410 | INFO     | tiberate.fhe.engine.ckks_engine:__init__:56 - CKKS parameters not specified. Using silver preset.
2025-03-18 11:33:35.443 | INFO     | tiberate.ntt.ntt_context:__init__:24 - Device not specified. Using default ['cuda:0'].
2025-03-18 11:33:35.636 | DEBUG    | tiberate.fhe.engine.ckks_engine:sk:87 - Created a new secret key.
2025-03-18 11:33:35.637 | DEBUG    | tiberate.fhe.engine.ckks_engine:pk:101 - Created a new public key.
2025-03-18 11:33:35.657 | DEBUG    | tiberate.fhe.engine.ckks_engine:evk:112 - Created a new evaluation key.
2025-03-18 11:33:35.666 | DEBUG    | tiberate.fhe.engine.ckks_engine:_create_rotation_key:1359 - Rotation key created for delta 1
2025-03-18 11:33:35.674 | DEBUG    | tiberate.fhe.engine.ckks_engine:_create_rotation_key:1359 - Rotation key created for delta -1
<class 'tiberate.fhe.engine.ckks_engine.CkksEngine'>
        Using NTT Context:
        <class 'tiberate.ntt.ntt_context.NTTContext'>
	        Using CKKS Context:
	        <class 'tiberate.fhe.context.ckks_context.CkksContext'>
		        buffer_bit_length = 62
		        scale_bits = 40
		        logN = 15
		        N = 32,768
		        Number of special primes = 2
		        Number of scales = 16
		        Cache folder = '/home/zgong6/repos/tiberate/tiberate/fhe/cache/resources'
		        Security bits = 128
		        Quantum security model = post_quantum
		        Security sampling distribution = uniform
		        Number of message bits = 60
		        Using '821' bits out of available maximum '829' bits.
		        Is Secured = True
		        RNS primes = [1099510054913, 1099515691009, 1099507695617, 1099516280833, 1099506515969, 1099520606209, 1099504549889, 1099523555329, 1099503894529, 1099527946241, 1099503370241, 1099529060353, 1099498258433, 1099531223041, 1099469684737, 1099532009473, 1152921504606584833, 1152921504598720513, 1152921504597016577].
	        Using devices = ['cuda:0']
	        Available levels = 17
	        Ordinary primes = 17
	        Special primes = 2


Plaintext(data=tensor([ 2.8118,  0.2544, -0.7268,  ..., -0.6502, -0.7935,  1.5616]), cached levels=[0, 1])
Mean: 5.629096068239243e-06, Std: 2.091297532980721e-05
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

## Rebuild

If you have modified CUDA or PyTorch versions, you may need to rebuild the package. You can do this by running:

```bash
tiberate rebuild
```

## Dev

See [csrc/readme.md](csrc/readme.md) for cpp configuration and build instructions.
