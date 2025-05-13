clean:
    find tiberate/libs -maxdepth 1 -type f -name "*.so" -exec rm -v {} \;
    echo "Cleaned up shared object files in tiberate/libs"

    find tiberate -type d -name "__pycache__" -exec rm -rv {} \;
    echo "Cleaned up __pycache__ directories in tiberate"

install:
    uv pip install -e . --no-build-isolation

trace:
    TORCH_LOGS="graph_breaks" tiberate benchmark
