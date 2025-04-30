clean:
    find tiberate/libs -maxdepth 1 -type f -name "*.so" -exec rm -v {} \;
    echo "Cleaned up shared object files in tiberate/libs"

dev:
    uv pip install -e . --no-build-isolation

trace:
    TORCH_LOGS="graph_breaks" tiberate benchmark
