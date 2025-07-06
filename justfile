clean-build: # .so in build directories, and build folder
    find tiberate -type d -name "build" -exec rm -rv {} \;
    find tiberate/libs -maxdepth 2 -type f -name "*.so" -exec rm -v {} \;
    echo "Cleaned up shared object files in tiberate/libs"

clean-cache: # all .pkl and .pt files
    find tiberate -type f \( -name "*.pkl" -o -name "*.pt" \) -exec rm -v {} \;

clean-pycache: # all __pycache__ directories
    find tiberate -type d -name "__pycache__" -exec rm -rv {} \;

clean-all: clean-build clean-cache clean-pycache

install:
    pip install --editable . --verbose --no-build-isolation

install-uv:
    uv pip install --editable . --verbose --no-build-isolation

trace:
    TORCH_LOGS="graph_breaks" tiberate benchmark
