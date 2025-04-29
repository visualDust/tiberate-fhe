#!/bin/bash

# This script removes all __pycache__ directories from the current directory and its subdirectories.

find . -type d -name '__pycache__' -exec rm -r {} + 2>/dev/null
echo "All __pycache__ directories have been removed."

find . -type f -name '*.so' -exec rm -f {} + 2>/dev/null
echo "All .so files have been removed."

find . -type f -name '*.pkl' -exec rm -f {} + 2>/dev/null
echo "All .pkl files have been removed."

find . -type f -name '*.pt' -exec rm -f {} + 2>/dev/null
echo "All .pt files have been removed."

find . -type d -name '.pytest_cache' -exec rm -r {} + 2>/dev/null
echo "All .pytest_cache directories have been removed."
