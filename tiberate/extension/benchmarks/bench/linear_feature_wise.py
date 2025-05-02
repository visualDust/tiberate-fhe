from time import time

import torch
from loguru import logger
from vdtoys.registry import Registry

from tiberate import CkksEngine, Preset

from ..components.interface import HELinear
from ..components.linear_feature_wise import (
    HELinear_FeatureWiseCTInput_ColMajorPTSquareWeight_FeatureWiseCTOutput,
)
from ..packing.feature_wise_compact import FeatureWise_CTEncoding
from ..packing.interface import CTEncoding
from .interface import (
    BenchmarkBase,
    BenchmarkResult,
    BenchmarkResultMetricType,
)

benchreg = Registry("benchmarks")


@benchreg.register(name="Linear Layer Evaluation with Feature-Wise Packing")
class LinearFeatureWiseBenchmark(BenchmarkBase):
    def __init__(self):
        self.name = "Linear Layer Evaluation with Feature-Wise Packing"
        self.description = "Evaluate the performance of a linear layer with feature-wise packing."

        self.config_matrix = {
            # (64, 256) x (256, 256)
            "(64,256)-logN14": {
                "description": "(64,256) input x (256,256) weight - polynomial degree logN14",
                "input_shape": (64, 256),
                "weight_shape": (256, 256),
                "ckks_params": Preset.logN14,
            },
            "(64,256)-logN15": {
                "description": "(64,256) input x (256,256) weight - polynomial degree logN15",
                "input_shape": (64, 256),
                "weight_shape": (256, 256),
                "ckks_params": Preset.logN15,
            },
            "(64,256)-logN16": {
                "description": "(64,256) input x (256,256) weight - polynomial degree logN16",
                "input_shape": (64, 256),
                "weight_shape": (256, 256),
                "ckks_params": Preset.logN16,
            },
            # (128, 512) x (512, 512)
            "(128,512)-logN14": {
                "description": "(128,512) input x (512,512) weight - polynomial degree logN14",
                "input_shape": (128, 512),
                "weight_shape": (512, 512),
                "ckks_params": Preset.logN14,
            },
            "(128,512)-logN15": {
                "description": "(128,512) input x (512,512) weight - polynomial degree logN15",
                "input_shape": (128, 512),
                "weight_shape": (512, 512),
                "ckks_params": Preset.logN15,
            },
            "(128,512)-logN16": {
                "description": "(128,512) input x (512,512) weight - polynomial degree logN16",
                "input_shape": (128, 512),
                "weight_shape": (512, 512),
                "ckks_params": Preset.logN16,
            },
            # (256, 1024) x (1024, 1024)
            "(256,1024)-logN14": {
                "description": "(256,1024) input x (1024,1024) weight - polynomial degree logN14",
                "input_shape": (256, 1024),
                "weight_shape": (1024, 1024),
                "ckks_params": Preset.logN14,
            },
            "(256,1024)-logN15": {
                "description": "(256,1024) input x (1024,1024) weight - polynomial degree logN15",
                "input_shape": (256, 1024),
                "weight_shape": (1024, 1024),
                "ckks_params": Preset.logN15,
            },
            "(256,1024)-logN16": {
                "description": "(256,1024) input x (1024,1024) weight - polynomial degree logN16",
                "input_shape": (256, 1024),
                "weight_shape": (1024, 1024),
                "ckks_params": Preset.logN16,
            },
        }

    def get_option_name2desc(self):
        return {k: v["description"] for k, v in self.config_matrix.items()}

    def run(self, option_name: str):
        assert (
            option_name in self.config_matrix
        ), f"Invalid benchmark name: {option_name}"
        config = self.config_matrix[option_name]
        input_shape = config["input_shape"]
        weight_shape = config["weight_shape"]
        ckks_params = config["ckks_params"]

        benchmark_result = BenchmarkResult()

        logger.info(f"Using config: {config['description']}")

        engine = CkksEngine(ckks_params)
        input_tensor = torch.randn(input_shape)
        input_tensor = input_tensor.unsqueeze(0)
        he_input = FeatureWise_CTEncoding.encodecrypt(
            src=input_tensor, engine=engine
        )
        torch_linear = torch.nn.Linear(
            in_features=weight_shape[0], out_features=weight_shape[1]
        )
        he_linear = HELinear_FeatureWiseCTInput_ColMajorPTSquareWeight_FeatureWiseCTOutput.fromTorch(
            torch_linear, engine
        )
        with torch.no_grad():
            torch_out = torch_linear(input_tensor)

        # Do the first time
        logger.info("Warm-up running...")
        he_out = he_linear(he_input.clone(), memory_save=True)
        # Do the second time
        logger.info("Running benchmark...")
        HELinear.debug = True
        CTEncoding.debug = True
        time0 = time()
        he_out = he_linear(he_input.clone(), memory_save=True)
        total_time = time() - time0

        benchmark_result.add_metric(
            name="Total Time (seconds)",
            metric_type=BenchmarkResultMetricType.SCALAR,
            series="default",
            value=total_time,
            description="Total time taken",
        )

        benchmark_result.add_metric(
            name="Throughput (samples/sec)",
            metric_type=BenchmarkResultMetricType.SCALAR,
            series="default",
            value=input_shape[0] / total_time,
            description="Samples per second, calculated as input.shape[0]/total_time",
        )

        dec_he_out = FeatureWise_CTEncoding.decryptcode(
            packed_ct=he_out, engine=engine, sk=engine.sk
        )
        diff = (torch_out[0] - dec_he_out[0]).float().detach()
        diff = diff.cpu().numpy()
        diff = diff.flatten()
        max_diff = diff.max()
        mean_diff = diff.mean()

        benchmark_result.add_metric(
            name="Max Difference",
            metric_type=BenchmarkResultMetricType.SCALAR,
            series="error",
            value=max_diff,
            description="Maximum difference between torch output and HE output",
        )
        benchmark_result.add_metric(
            name="Mean Difference",
            metric_type=BenchmarkResultMetricType.SCALAR,
            series="error",
            value=mean_diff,
            description="Mean difference between torch output and HE output",
        )

        return benchmark_result


if __name__ == "__main__":
    benchmark = LinearFeatureWiseBenchmark()
    benchmark.run("(64,256)-logN14")
