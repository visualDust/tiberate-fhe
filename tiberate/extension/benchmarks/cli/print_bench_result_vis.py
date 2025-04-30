import plotext
from rich.console import Console
from rich.table import Table

console = Console()
from ..bench.interface import BenchmarkResult, BenchmarkResultMetricType

# format of metrics, see @BenchmarkResult.add_metric
# {
#     metric_type(BenchmarkResultMetricType): {
#         series_1(str):[
#             {
#                 "name": str,
#                 "value": any,
#                 "description": str | None
#             },
#           ...
#          ],
#         series_2(str):[
#             {
#                 "name": str,
#                 "value": any,
#                 "description": str | None
#             },
#           ...
#         ],
#     }
# }


def _handle_plot(metrics: dict):
    for series, metrics_list in metrics.items():
        descriptions = {}
        for metric in metrics_list:
            name = metric["name"]
            value = metric["value"]
            description = metric.get("description", "")
            plotext.plot(value, label=name)
            if series == "default":
                plotext.title(name)
                plotext.show()
                console.print(
                    f"[bold]{name}[/bold]: {description}", style="dim"
                )
            else:
                descriptions[name] = description
        plotext.title(series)
        plotext.show()
        # print descriptions
        if descriptions:
            console.print("[bold]Figure note[/bold]:", style="dim", end="")
            for name, description in descriptions.items():
                console.print(f"{name}: {description}", style="dim", end="; ")
            print()  # for a new line after all descriptions


def _handle_table(metrics: dict):
    for series, metrics_list in metrics.items():
        for metric in metrics_list:
            name = metric["name"]
            value = metric["value"]  # should be a 2D array or similar structure
            description = metric.get("description", "")
            table = Table(
                title=f"{series} - " if series != "default" else "" + f"{name}"
            )
            # first row is the header
            for header in value[0]:
                table.add_column(header)
            # add the rest of the rows
            for row in value[1:]:
                table.add_row(*[str(cell) for cell in row])
            console.print(table)
            if description:
                # Print the description as a note
                console.print(
                    f"[bold]Table note[/bold]: {description}", style="dim"
                )


def _handle_scalar(metrics: dict):
    # handle as table, each metric is a single value
    table = Table(title="Scalar Metrics")
    table.add_column("Name")
    table.add_column("Value")
    table.add_column("Description", justify="right")
    for series, metrics_list in metrics.items():
        for metric in metrics_list:
            name = (
                f"{series} - {metric['name']}"
                if series != "default"
                else metric["name"]
            )
            value = metric["value"]
            description = metric.get("description", "")
            table.add_row(name, str(value), description)
    console.print(table)


def _handle_distribution(metrics: dict):
    raise NotImplementedError(
        "Distribution metrics visualization is not implemented yet."
    )
    # # same as plot, but use plotext.hist
    # for series, metrics_list in metrics.items():
    #     descriptions = {}
    #     for metric in metrics_list:
    #         name = metric["name"]
    #         value = metric["value"]
    #         description = metric.get("description", "")
    #         print(value)
    #         plotext.hist(value, bins=50, label=name)
    #         if series == "default":
    #             plotext.title(name)
    #             plotext.show()
    #             console.print(
    #                 f"[bold]{name}[/bold]: {description}", style="dim"
    #             )
    #         else:
    #             descriptions[name] = description
    #     plotext.title(series)
    #     plotext.show()
    #     # print descriptions
    #     if descriptions:
    #         console.print("[bold]Figure note[/bold]:", style="dim", end="")
    #         for name, description in descriptions.items():
    #             console.print(f"{name}: {description}", style="dim", end="; ")
    #         print()  # for a new line after all descriptions


METRIC_TYPE_2_HANDLER = {
    BenchmarkResultMetricType.SCALAR: _handle_scalar,
    BenchmarkResultMetricType.PLOT: _handle_plot,
    BenchmarkResultMetricType.TABLE: _handle_table,
    BenchmarkResultMetricType.DISTRIBUTION: _handle_distribution,
}


def visualize_benchmark_result(benchmark_result: BenchmarkResult):
    for metric_type, metrics in benchmark_result.metrics.items():
        if metric_type not in METRIC_TYPE_2_HANDLER:
            console.print(
                f"[bold red]Unsupported metric type: {metric_type}[/bold red]"
            )
            continue
        handler = METRIC_TYPE_2_HANDLER[metric_type]
        handler(metrics)
