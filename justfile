clean:
    find tiberate -type d -name "__pycache__" -exec rm -rv {} \;
    echo "Cleaned up __pycache__ directories in tiberate"

clean-build:
    find tiberate/libs -maxdepth 99 -type f -name "*.so" -exec rm -v {} \;
    echo "Cleaned up shared object files in tiberate/libs"

install:
    uv pip install -e . --no-build-isolation

trace:
    TORCH_LOGS="graph_breaks" tiberate benchmark
