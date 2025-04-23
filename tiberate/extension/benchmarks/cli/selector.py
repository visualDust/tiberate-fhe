import time
from typing import Dict

from loguru import logger
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.events import Key
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
)

from .. import BenchmarkBase
from ..bench import benchreg


class BenchSelector(App):
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+r", "run_bench", "Run Benchmark"),
    ]
    CSS_PATH = "bench_selector.tcss"

    focus_on_bench = reactive(True)

    def __init__(self, benches: Dict[str, BenchmarkBase], **kwargs):
        super().__init__(**kwargs)
        self.benches = benches
        self.bench_names = list(self.benches.keys())
        self.current_bench: BenchmarkBase | None = None
        self.current_options: Dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_area"):
            with Container(id="left_container"):
                yield Static("Select Benchmark", id="bench_label")
                yield ListView(id="bench_list")
                yield Markdown(
                    "",
                    id="bench_description",
                    classes="bottom-aligned",
                )

            with Container(id="right_container"):
                # left and right container for label and button
                yield Static("Select Option", id="option_label")
                yield ListView(id="option_list")
                yield Markdown(
                    "",
                    id="option_description",
                    classes="bottom-aligned",
                )

        yield Footer()

    def on_mount(self) -> None:
        self.title = "Tiberate Benchmarks"
        self.sub_title = "Select a benchmark to run"

        self.update_bench_list()
        if len(self.bench_names) == 1:
            self.query_one("#bench_list", ListView).index = 0
            self.select_benchmark()
        self.update_highlight()
        self.set_focus(self.query_one("#bench_list", ListView))

    def update_highlight(self) -> None:
        left = self.query_one("#left_container", Container)
        right = self.query_one("#right_container", Container)
        if self.focus_on_bench:
            left.add_class("highlight")
            right.remove_class("highlight")
        else:
            right.add_class("highlight")
            left.remove_class("highlight")

    def update_bench_list(self) -> None:
        bench_list = self.query_one("#bench_list", ListView)
        bench_list.clear()

        for name in self.bench_names:
            bench_list.append(ListItem(Label(name)))

        if self.bench_names:
            bench_list.index = 0  # Default to first item

            selected_name = self.bench_names[0]
            description = self.benches[selected_name].description
            self.query_one("#bench_description", Markdown).update(str(description))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id == "bench_list":
            self.focus_on_bench = True
            self.set_focus(event.list_view)
            self.select_benchmark()  # Use ListView.index

        elif event.list_view.id == "option_list":
            self.focus_on_bench = False
            self.set_focus(event.list_view)
            self.select_bench_options()  # Use ListView.index

    def update_option_list(self) -> None:
        option_list = self.query_one("#option_list", ListView)
        option_list.clear()
        option_desc = self.query_one("#option_description", Markdown)

        if self.current_options:
            for name in self.current_options:
                option_list.append(ListItem(Label(name)))

            option_list.index = 0  # Always default to first option
            self.select_bench_options()  # Sync description with selection
        else:
            option_desc.update("No options available for this benchmark.")

    def select_benchmark(self) -> None:
        list_view = self.query_one("#bench_list", ListView)
        index = list_view.index
        if 0 <= index < len(self.bench_names):
            name = self.bench_names[index]
            self.current_bench = self.benches[name]
            self.current_options = self.current_bench.get_bench_option2desc()

            # Update right panel
            self.update_option_list()

            # Update left description
            description = self.benches[name].description
            self.query_one("#bench_description", Markdown).update(str(description))

    def select_bench_options(self) -> None:
        list_view = self.query_one("#option_list", ListView)
        index = list_view.index

        if self.current_options and 0 <= index < len(self.current_options):
            selected_name = list(self.current_options.keys())[index]
            self.query_one("#option_description", Markdown).update(
                str(self.current_options[selected_name])
            )

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id == "bench_list" and self.focus_on_bench:
            self.select_benchmark()
        elif event.list_view.id == "option_list" and not self.focus_on_bench:
            self.select_bench_options()

    def on_key(self, event: Key) -> None:
        key = event.key
        if key == "left" and not self.focus_on_bench:
            self.focus_on_bench = True
            self.set_focus(self.query_one("#bench_list", ListView))
        elif key == "right" and self.focus_on_bench:
            self.focus_on_bench = False
            self.set_focus(self.query_one("#option_list", ListView))
            option_list = self.query_one("#option_list", ListView)
            option_list.index = 0
            self.select_bench_options()

    def action_run_bench(self) -> None:
        if self.focus_on_bench:
            self.select_benchmark()
        else:
            self.set_run_benchmark()

    def set_run_benchmark(self) -> None:
        if not self.current_bench:
            self.console.print("[bold red]No benchmark selected.[/bold red]")
            return

        # Save info to run after exit
        self._should_run_bench = True
        self._selected_bench = self.current_bench
        self._selected_option = (
            "default"
            if not self.current_options
            else list(self.current_options.keys())[self.query_one("#option_list", ListView).index]
        )

        self.exit()


def main():
    benches = {name: benchreg[name] for name in benchreg.keys()}
    available_bench_instances = {}
    for name, benchCls in benches.items():
        try:
            bench_instance = benchCls()
            available_bench_instances[name] = bench_instance
            logger.success(f"Created benchmark instance {name}")
        except Exception as e:
            logger.error(f"Failed to initialize benchmark {name}: {e}")
            continue

    time.sleep(0.5)

    app = BenchSelector(available_bench_instances)
    app._should_run_bench = False
    app._selected_bench = None
    app._selected_option = None
    app.run()

    if app._should_run_bench and app._selected_bench:
        logger.info(
            f"Running Benchmark: {app._selected_bench.name} | Option: {app._selected_option}]"
        )
        app._selected_bench.run(app._selected_option)


if __name__ == "__main__":
    main()
