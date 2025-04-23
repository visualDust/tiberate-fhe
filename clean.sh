#!/bin/bash

# This script removes all __pycache__ directories from the current directory and its subdirectories.

echo "Searching for __pycache__ directories to remove..."
find . -type d -name '__pycache__' -exec rm -r {} + 2>/dev/null
echo "All __pycache__ directories have been removed."

echo "Searching for .so files to remove..."
find . -type f -name '*.so' -exec rm -f {} + 2>/dev/null
echo "All .so files have been removed."

echo "Removing .pytest_cache directories..."
find . -type d -name '.pytest_cache' -exec rm -r {} + 2>/dev/null
echo "All .pytest_cache directories have been removed."
