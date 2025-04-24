import time

import torch
from loguru import logger
from vdtoys.registry import Registry

from tiberate import CkksEngine, presets

from .interface import BenchmarkBase

benchreg = Registry("benchmarks")


@benchreg.register(name="CMult Single OP")
class CMultSingleOPBenchmark(BenchmarkBase):
    def __init__(self):
        self.name = "CMult Single OP"
        self.description = "Evaluate the performance of a single Ciphertext-Ciphertext multiplication operation. Latency is averaged over 100 runs."
        self.config_matrix = {
            "logN14 No Relinearize": {
                "description": "Using polynomial degree logN14, run CMult(without relinearization) for 100 times",
                "ckks_params": presets.logN14,
                "relinearize": False,
            },
            "logN14": {
                "description": "Using polynomial degree logN14, run CMult for 100 times",
                "ckks_params": presets.logN14,
                "relinearize": True,
            },
            "logN15 No Relinearize": {
                "description": "Using polynomial degree logN15, run CMult(without relinearization) for 100 times",
                "ckks_params": presets.logN15,
                "relinearize": False,
            },
            "logN15": {
                "description": "Using polynomial degree logN15, run CMult for 100 times",
                "ckks_params": presets.logN15,
                "relinearize": True,
            },
            "logN16 No Relinearize": {
                "description": "Using polynomial degree logN16, run CMult(without relinearization) for 100 times",
                "ckks_params": presets.logN16,
                "relinearize": False,
            },
            "logN16": {
                "description": "Using polynomial degree logN16, run CMult for 100 times",
                "ckks_params": presets.logN16,
                "relinearize": True,
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

        engine = CkksEngine(ckks_params=ckks_params)
        input_tensor_1 = torch.randn((engine.num_slots,))
        input_tensor_2 = torch.randn((engine.num_slots,))
        plain_output = input_tensor_1 * input_tensor_2

        # Encrypt the input tensors
        packed_ct_1 = engine.encodecrypt(input_tensor_1)
        packed_ct_2 = engine.encodecrypt(input_tensor_2)

        # Perform the multiplication
        ct_out = engine.cc_mult(
            packed_ct_1, packed_ct_2, post_relin=config["relinearize"]
        )

        time0 = time.time()
        for _ in range(100):
            ct_out = engine.cc_mult(
                packed_ct_1, packed_ct_2, post_relin=config["relinearize"]
            )
        time1 = time.time()

        dec_he_out = engine.decryptcode(ct_out)
        diff = (plain_output - dec_he_out).float().detach()
        diff = diff.cpu().numpy()
        diff = diff.flatten()
        max_diff = diff.max()
        mean_diff = diff.mean()

        latency = (time1 - time0) / 100 * 1000  # Convert to milliseconds

        logger.info(
            f"Max diff: {max_diff}, Mean diff: {mean_diff}, Latency: {latency:.4f} milliseconds"
        )

        logger.info(f"Benchmark {self.name} completed successfully.")
