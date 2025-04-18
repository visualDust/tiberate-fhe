# setup cpp env for vscode

## setup project

See [../README.md](../README.md) for instructions on how to set up the project.

## enter conda env and install cmake

```bash
conda activate <your_env>
conda install cmake python-devtools
```

## generate compile_commands.json

```bash
mkdir build
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
```

## configure vscode

### install and configure clangd extension

1. Install [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) extension in vscode. Using local clangd-20 is recommended, one known issue is that using clangd-14 (default on Ubuntu 22.04) will cause CUDA device code does not support variadic functions.
2. Set `--compile-commands-dir`: in vscode clangd settings, add `--compile-commands-dir=${workspaceFolder}/csrc/build`
