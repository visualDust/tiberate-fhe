# setup cpp env for vscode

## setup project

See [../README.md](../README.md) for instructions on how to set up the project.

## enter conda env and install cmake

```bash
conda activate <your_env>
conda install cmake
```

## softlink torch include dir

```bash
ln -s $(python -c "import torch; import os; print(os.path.abspath(os.path.join(torch.utils.cmake_prefix_path, '../../')))") ./libtorch
```

## generate compile_commands.json

```bash
mkdir build
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
```

## configure vscode

### install and configure clangd extension

1. Install [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) extension in vscode
2. Set `--compile-commands-dir`: in vscode clangd settings, add `--compile-commands-dir=${workspaceFolder}/csrc/build`
