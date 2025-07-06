import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from vdtoys.localstorage import format_bytes

console = Console()


@click.group()
@click.pass_context
def options(ctx):
    ctx.ensure_object(dict)


@options.command(name="version")
def version_command():
    """Print the version of tiberate"""
    import tiberate

    console.print(
        Panel(
            f"tiberate version: {tiberate.__version__}",
            title="Version",
            title_align="left",
            border_style="cyan",
        )
    )


@options.command(name="benchmark")
@click.option(
    "--file",
    required=False,
    help="Path to a benchmark file to run, If not provided, opens the benchmark selector.",
)
def benchmark_command(file: str | None = None):
    """Open benchmark page or run a benchmark file"""
    if file:
        file_path = os.path.abspath(file)
        if not os.path.isfile(file_path):
            click.echo(f"[Error] File not found: {file_path}", err=True)
            raise click.Abort
        click.echo(f"[Info] Running benchmark file: {file_path}")
        # import the benchmark file dynamically and then launch benchselector
        import importlib.util

        spec = importlib.util.spec_from_file_location("benchmark", file_path)
        if spec is None:
            click.echo(
                f"[Error] Could not load benchmark file: {file_path}", err=True
            )
            raise click.Abort
        benchmark_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark_module)

    # Run the benchmark selector
    from tiberate.extension.benchmarks.cli.selector import (
        main as bench_selector,
    )

    bench_selector()


@options.command(name="devices")
def show_devices():
    """Show CUDA devices information"""
    from tiberate.libs.wrapper.cuda_info import get_cuda_info

    def display_device_info(info_dict):
        for device_id, props in info_dict.items():
            table = Table.grid(padding=(0, 1))  # No border, just spacing
            table.add_column("Property", style="cyan", no_wrap=True)
            table.add_column("Value", style="white")

            for key, value in props.items():
                if key == "name":
                    continue  # Skip the name property, handled in the title
                if key in {
                    "totalGlobalMem",
                    "sharedMemPerBlock",
                    "sharedMemPerMultiprocessor",
                    "totalConstMem",
                    "l2CacheSize",
                }:
                    display_value = format_bytes(value)
                else:
                    display_value = Pretty(value, max_length=100)

                table.add_row(key, display_value)

            console.print(
                Panel(
                    table,
                    title=f"[bold yellow]CUDA Device {device_id}[/] - [green]{props.get('name', '')}[/]",
                    border_style="magenta",
                    expand=False,
                )
            )

    info = None
    try:
        info = get_cuda_info()
    except Exception as e:
        console.print(
            f"[red]Error retrieving CUDA device information: {e}[/red]"
        )
        return
    if not info:
        console.print("[red]No CUDA devices found.[/red]")
    else:
        display_device_info(info)


# @main.command(name="rebuild", context_settings={"max_content_width": 120})
# def rebuild_command():
#     """Rebuild CSRC with setup.py after you modified CUDA or PyTorch version, or when you change the csrc code."""
#     # Ask for confirmation before rebuilding
#     if click.confirm("Are you sure you want to rebuild the NTT backend?", default=False):
#         # the setup.py is in the root directory of tiberate/..
#         # so we need to change the working directory to the root directory
#         from tiberate import __file__ as tiberate_path

#         tiberate_dir = os.path.dirname(os.path.dirname(tiberate_path))
#         os.chdir(tiberate_dir)
#         os.system("python setup.py build")
#     else:
#         console.print("[red]Rebuild canceled.[/red]")


if __name__ == "__main__":
    options()
