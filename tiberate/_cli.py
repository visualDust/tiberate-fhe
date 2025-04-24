import os

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
@click.pass_context
def main(ctx):
    ctx.ensure_object(dict)


@main.command(name="version")
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


@main.command(name="benchmark")
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
    main()
