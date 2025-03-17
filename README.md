# What is this

this is a variant of [Desilo/liberate-fhe](https://github.com/Desilo/liberate-fhe) with better typing (and for my own modifications). This fork has left fork network and will not return to the original branch. This might not be a drop-in replacement for the original one, but it should be compatible with most of the code.

> [!CAUTION]
> This project is still in development, and the API may change in the future. Please use it with caution.

## Pre-requisites

- Python 3.10 or later, Python 3.12 is recommended
- CUDA 11.8 or later, CUDA 12+ is recommended

> [!IMPORTANT]
> This is a GPU only library, and it will not work or even compile on CPU only machines.

## Install

![PyPI - Version](https://img.shields.io/pypi/v/tiberate) ![PyPI - Downloads](https://img.shields.io/pypi/dw/tiberate)

```bash
pip install tiberate --verbose
```

> [!NOTE]
> If you encounter the error saying "The detected CUDA version mismatches the version that was used to compile PyTorch.", you can switch cuda version, or manually install pytorch with the same CUDA version as your system, then install this package without dependencies via:
>
> ```bash
> pip install tiberate --no-deps --verbose
> ```

## Install from source

```bash
# clone the repository
git clone https://github.com/wens-lab/tiberate-fhe.git
cd tiberate-fhe
# install the dependencies
poetry install # this will also install the package itself into current environment
```
