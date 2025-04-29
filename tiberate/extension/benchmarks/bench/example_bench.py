import os

from loguru import logger
from vdtoys.registry import Registry

from tiberate.config import Preset

from .interface import BenchmarkBase

benchreg = Registry("benchmarks")


# @benchreg.register(name="Benchmark Example Code")
class ExampleBenchClass(BenchmarkBase):
    def __init__(self):
        super().__init__()
        self.name = "Example Benchmark"

        file_path = os.path.dirname(__file__)
        readme_path = os.path.join(file_path, "readme.md")
        with open(readme_path, "r") as f:
            self.description = f.read()

        self.config_matrix = {
            # (64, 256) x (256, 256)
            "Example-64-logN14": {
                "description": "(64,256) input x (256,256) weight - polynomial degree logN14",
                "input_shape": (64, 256),
                "weight_shape": (256, 256),
                "ckks_params": Preset.logN14,
            },
            "Example-64-logN15": {
                "description": "(64,256) input x (256,256) weight - polynomial degree logN15",
                "input_shape": (64, 256),
                "weight_shape": (256, 256),
                "ckks_params": Preset.logN15,
            },
            "Example-64-logN16": {
                "description": "(64,256) input x (256,256) weight - polynomial degree logN16",
                "input_shape": (64, 256),
                "weight_shape": (256, 256),
                "ckks_params": Preset.logN16,
            },
            # (128, 512) x (512, 512)
            "Example-128-logN14": {
                "description": "(128,512) input x (512,512) weight - polynomial degree logN14",
                "input_shape": (128, 512),
                "weight_shape": (512, 512),
                "ckks_params": Preset.logN14,
            },
            "Example-128-logN15": {
                "description": "(128,512) input x (512,512) weight - polynomial degree logN15",
                "input_shape": (128, 512),
                "weight_shape": (512, 512),
                "ckks_params": Preset.logN15,
            },
            "Example-128-logN16": {
                "description": "(128,512) input x (512,512) weight - polynomial degree logN16",
                "input_shape": (128, 512),
                "weight_shape": (512, 512),
                "ckks_params": Preset.logN16,
            },
            # (256, 1024) x (1024, 1024)
            "Example-256-logN14": {
                "description": "(256,1024) input x (1024,1024) weight - polynomial degree logN14",
                "input_shape": (256, 1024),
                "weight_shape": (1024, 1024),
                "ckks_params": Preset.logN14,
            },
            "Example-256-logN15": {
                "description": "(256,1024) input x (1024,1024) weight - polynomial degree logN15",
                "input_shape": (256, 1024),
                "weight_shape": (1024, 1024),
                "ckks_params": Preset.logN15,
            },
            "Example-256-logN16": {
                "description": "(256,1024) input x (1024,1024) weight - polynomial degree logN16",
                "input_shape": (256, 1024),
                "weight_shape": (1024, 1024),
                "ckks_params": Preset.logN16,
            },
            # (512, 2048) x (2048, 2048)
            "Example-512-logN14": {
                "description": "(512,2048) input x (2048,2048) weight - polynomial degree logN14",
                "input_shape": (512, 2048),
                "weight_shape": (2048, 2048),
                "ckks_params": Preset.logN14,
            },
            "Example-512-logN15": {
                "description": "(512,2048) input x (2048,2048) weight - polynomial degree logN15",
                "input_shape": (512, 2048),
                "weight_shape": (2048, 2048),
                "ckks_params": Preset.logN15,
            },
            "Example-512-logN16": {
                "description": "(512,2048) input x (2048,2048) weight - polynomial degree logN16",
                "input_shape": (512, 2048),
                "weight_shape": (2048, 2048),
                "ckks_params": Preset.logN16,
            },
        }

    def get_bench_option2desc(self):
        return {
            name: config["description"]
            for name, config in self.config_matrix.items()
        }

    def run(self, option_name):
        logger.info(f"Running benchmark with option: {option_name}")
        logger.info(f"Configuration: {self.config_matrix[option_name]}")

        logger.warning(
            "This is an example benchmark. The actual implementation of the run method is not provided. So I will raise NotImplementedError here."
        )
        raise NotImplementedError("Run method not implemented yet.")
