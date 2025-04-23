class BenchmarkBase:
    """
    A class to represent a benchmark.

    Attributes
    ----------
    name : str
        The name of the benchmark.
    description : str
        A brief description of the benchmark.
    version : str
        The version of the benchmark.

    Please also decorate the class with:
    ```python
    from vdtoys.registry import Registry
    benchreg = Registry("benchmarks")
    @benchreg.register(name="your bench mark name")
    class YourBenchmark(Benchmark):
        def gen_bench_options(self):
            return {
                "name1": "description1",
                "name2": "description2",
                ...
            }

        def run(self, name: str):
            # Implement the benchmark logic here
            ...
    ```
    So that the benchmark cli can discover it.
    """

    name: str  # The name of the benchmark
    description: str  # A brief description of the benchmark

    def __init__(self):
        # The benchmark should have name and description attributes.
        self.name = "Example Benchmark"
        self.description = """
# Example Benchmark
This is an example benchmark class that demonstrates how to implement a benchmark in Tiberate.

## Usage

It should be decorated with the `@benchreg.register(name="your bench mark name")` decorator to be discoverable by the benchmark CLI.
"""

    def get_bench_option2desc(self):
        """
        Should return a dict with string keys and values, like:
        {
            "name1": "description1",
            "name2": "description2",
            ...
        }
        The description can be markedown formatted, but it is not required.
        Those keys are the names that can be passed to the benchmark.
        """
        raise NotImplementedError("gen_bench_options() must be implemented in subclasses")

    def run(self, option_name: str) -> dict:
        """
        Should run the benchmark with the given name.

        Returns a dictionary with benchmark results. You can also print the results directly
        to the console if you prefer. The dictionary should contain relevant metrics and results
        of the benchmark run.

        Generally, there are some recommended metrics to return:
        - Time taken, in seconds
        - Throughput, such as tokens per second
        - Error related metrics, such as (Max) difference between expected and actual results
        - Memory usage (maybe peak memory usage)
        - Number of operations (if applicable)

        Any other relevant metrics specific to the benchmark should also be returned.

        """
        raise NotImplementedError("run() must be implemented in subclasses")
