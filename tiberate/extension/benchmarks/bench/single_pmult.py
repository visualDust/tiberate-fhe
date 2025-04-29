import time

import torch
from loguru import logger
from vdtoys.registry import Registry

from tiberate import CkksConfig, CkksEngine, Preset
from tiberate.typing import Plaintext

from .interface import BenchmarkBase

benchreg = Registry("benchmarks")


@benchreg.register(name="PMult Single OP")
class PMultSingleOPBenchmark(BenchmarkBase):
    def __init__(self):
        self.name = "PMult Single OP"
        self.description = "Evaluate the performance of a single Plaintext-Ciphertext multiplication operation. Latency is averaged over 100 runs."
        self.config_matrix = {
            "logN14": {
                "description": "Using polynomial degree logN14, run PMult for 100 times",
                "ckks_params": Preset.logN14,
            },
            "logN15": {
                "description": "Using polynomial degree logN15, run PMult for 100 times",
                "ckks_params": Preset.logN15,
            },
            "logN16": {
                "description": "Using polynomial degree logN16, run PMult for 100 times",
                "ckks_params": Preset.logN16,
            },
        }

    def get_bench_option2desc(self):
        return {k: v["description"] for k, v in self.config_matrix.items()}

    def run(self, option_name):
        assert (
            option_name in self.config_matrix
        ), f"Invalid benchmark name: {option_name}"
        config = self.config_matrix[option_name]
        ckks_params = config["ckks_params"]
        logger.info(f"Running benchmark: {config['description']}")

        engine = CkksEngine(ckks_config=CkksConfig.from_preset(ckks_params))
        input_tensor_1 = torch.randn((engine.num_slots,))
        input_tensor_2 = torch.randn((engine.num_slots,))
        plain_output = input_tensor_1 * input_tensor_2

        # Encrypt the input tensors
        packed_ct_1 = engine.encodecrypt(input_tensor_1)
        pt = Plaintext(input_tensor_2)

        # Perform the multiplication
        ct_out = engine.pc_mult(pt=pt, ct=packed_ct_1)  # warmup

        time0 = time.time()
        for _ in range(100):
            ct_out = engine.pc_mult(pt=pt, ct=packed_ct_1)
        time1 = time.time()

        dec_he_out = engine.decryptcode(ct_out)
        diff = (plain_output - dec_he_out).float().detach()
        diff = diff.cpu().numpy()
        diff = diff.flatten()
        max_diff = diff.max()
        mean_diff = diff.mean()

        latency = (time1 - time0) / 100 * 1000  # in ms

        logger.info(
            f"Max diff: {max_diff}, Mean diff: {mean_diff}, Latency: {latency:.4f} milliseconds"
        )

        logger.info(f"Benchmark {self.name} completed successfully.")


if __name__ == "__main__":
    benchmark = PMultSingleOPBenchmark()
    benchmark.run("logN14")  # Example run with logN14 configuration
