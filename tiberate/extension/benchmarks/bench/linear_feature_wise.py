import torch
from loguru import logger
from vdtoys.registry import Registry

from tiberate import CkksEngine, presets

from ..components.interface import HELinear
from ..components.linear_feature_wise import (
    HELinear_FeatureWiseCTInput_ColMajorPTSquareWeight_FeatureWiseCTOutput,
)
from ..packing.feature_wise_compact import FeatureWise_CTEncoding
from ..packing.interface import CTEncoding
from .interface import BenchmarkBase

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
                "ckks_params": presets.logN14,
            },
            "(64,256)-logN15": {
                "description": "(64,256) input x (256,256) weight - polynomial degree logN15",
                "input_shape": (64, 256),
                "weight_shape": (256, 256),
                "ckks_params": presets.logN15,
            },
            "(64,256)-logN16": {
                "description": "(64,256) input x (256,256) weight - polynomial degree logN16",
                "input_shape": (64, 256),
                "weight_shape": (256, 256),
                "ckks_params": presets.logN16,
            },
            # (128, 512) x (512, 512)
            "(128,512)-logN14": {
                "description": "(128,512) input x (512,512) weight - polynomial degree logN14",
                "input_shape": (128, 512),
                "weight_shape": (512, 512),
                "ckks_params": presets.logN14,
            },
            "(128,512)-logN15": {
                "description": "(128,512) input x (512,512) weight - polynomial degree logN15",
                "input_shape": (128, 512),
                "weight_shape": (512, 512),
                "ckks_params": presets.logN15,
            },
            "(128,512)-logN16": {
                "description": "(128,512) input x (512,512) weight - polynomial degree logN16",
                "input_shape": (128, 512),
                "weight_shape": (512, 512),
                "ckks_params": presets.logN16,
            },
            # (256, 1024) x (1024, 1024)
            "(256,1024)-logN14": {
                "description": "(256,1024) input x (1024,1024) weight - polynomial degree logN14",
                "input_shape": (256, 1024),
                "weight_shape": (1024, 1024),
                "ckks_params": presets.logN14,
            },
            "(256,1024)-logN15": {
                "description": "(256,1024) input x (1024,1024) weight - polynomial degree logN15",
                "input_shape": (256, 1024),
                "weight_shape": (1024, 1024),
                "ckks_params": presets.logN15,
            },
            "(256,1024)-logN16": {
                "description": "(256,1024) input x (1024,1024) weight - polynomial degree logN16",
                "input_shape": (256, 1024),
                "weight_shape": (1024, 1024),
                "ckks_params": presets.logN16,
            },
        }

    def get_bench_option2desc(self):
        return {k: v["description"] for k, v in self.config_matrix.items()}

    def run(self, name: str):
        assert name in self.config_matrix, f"Invalid benchmark name: {name}"
        config = self.config_matrix[name]
        input_shape = config["input_shape"]
        weight_shape = config["weight_shape"]
        ckks_params = config["ckks_params"]
        logger.info(f"Running benchmark: {config['description']}")

        CTEncoding.debug = True
        HELinear.debug = True

        engine = CkksEngine(ckks_params=ckks_params)
        input_tensor = torch.randn(input_shape)
        input_tensor = input_tensor.unsqueeze(0)
        he_input = FeatureWise_CTEncoding.encodecrypt(src=input_tensor, engine=engine)
        torch_linear = torch.nn.Linear(in_features=weight_shape[0], out_features=weight_shape[1])
        he_linear = (
            HELinear_FeatureWiseCTInput_ColMajorPTSquareWeight_FeatureWiseCTOutput.fromTorch(
                torch_linear, engine
            )
        )
        with torch.no_grad():
            torch_out = torch_linear(input_tensor)
        he_out = he_linear(he_input, memory_save=True)
        dec_he_out = FeatureWise_CTEncoding.decryptcode(
            packed_ct=he_out, engine=engine, sk=engine.sk
        )
        diff = (torch_out[0] - dec_he_out[0]).float().detach()
        diff = diff.cpu().numpy()
        diff = diff.flatten()
        max_diff = diff.max()
        mean_diff = diff.mean()

        logger.info(
            f"Max diff: {max_diff}, Mean diff: {mean_diff}, "
            f"Input shape: {input_shape}, Weight shape: {weight_shape}"
        )
        logger.info(f"Benchmark {config['description']} completed successfully.")


if __name__ == "__main__":
    benchmark = LinearFeatureWiseBenchmark()
    benchmark.run("(64,256)-logN14")
