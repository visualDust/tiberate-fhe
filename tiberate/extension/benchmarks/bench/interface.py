from collections import defaultdict
from enum import Enum


class BenchmarkResultMetricType(Enum):
    SCALAR = "scalar"  # A single value metric
    PLOT = "plot"  # A list of values, such as a time series
    TABLE = (
        "table"  # A 2D list/array, the first row will be treated as the header
    )
    DISTRIBUTION = "distribution"  # A list of values, such as a histogram


class BenchmarkResult:
    """
    A class to represent the result of a benchmark run.

    Attributes
    ----------
    metrics : dict
        A dictionary containing the metrics of the benchmark run.
    misc : dict
        A dictionary containing any miscellaneous information related to the benchmark run.
    errors : dict
        A dictionary containing any errors encountered during the benchmark run.
    warnings : dict
        A dictionary containing any warnings encountered during the benchmark run.
    """

    def __init__(self):
        self.metrics = defaultdict(dict)
        self.misc = {}
        self.errors = {}
        self.warnings = {}

    def add_metric(
        self,
        name: str,
        metric_type: BenchmarkResultMetricType,
        value: any,
        series: str | None = None,
        description: str | None = None,
    ):
        """
        Add a metric to the benchmark result.

        Parameters
        ----------
        name : str
            The name of the metric, should be unique within the series.
        series : str
            The series of the metric, metrics with the same series will be grouped together.
        metric_type : BenchmarkResultMetricType
            The type of the metric (scalar, table, pairwise, distribution).
        value : any
            The value of the metric.
        description : str | None
            A description of the metric (optional).
        """
        if not isinstance(metric_type, BenchmarkResultMetricType):
            raise ValueError(
                f"Invalid metric type: {metric_type}. Must be one of {list(BenchmarkResultMetricType)}"
            )

        if series is None:
            series = "default"

        if metric_type not in self.metrics:
            self.metrics[metric_type] = {}

        if series not in self.metrics[metric_type]:
            self.metrics[metric_type][series] = []

        self.metrics[metric_type][series].append(
            {
                "name": name,
                "value": value,
                "description": description,
            }
        )

    def __repr__(self):
        return f"BenchmarkResult(metrics={self.metrics}, misc={self.misc}, errors={self.errors}, warnings={self.warnings})"


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

    def get_option_name2desc(self):
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
        raise NotImplementedError(
            "gen_bench_options() must be implemented in subclasses"
        )

    def run(self, option_name: str) -> BenchmarkResult:
        """
        Should run the benchmark with the given name.

        Returns a BenchmarkResult object.
        The result should contain the metrics of the benchmark run.
        The metrics should be added using the `add_metric` method of the @BenchmarkResult class.

        Generally, there are some recommended metrics to return:
        - Time taken, in seconds
        - Throughput, such as tokens per second
        - Error related metrics, such as (Max) difference between expected and actual results
        - Memory usage (maybe peak memory usage)
        - Number of operations (if applicable)

        Any other relevant metrics specific to the benchmark should also be returned.

        """
        raise NotImplementedError("run() must be implemented in subclasses")
