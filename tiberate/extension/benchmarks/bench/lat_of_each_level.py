import datetime
import time

import click
import torch
from loguru import logger
from vdtoys.ansi import legal_file_name_of
from vdtoys.registry import Registry

from tiberate import CkksEngine, Preset, errors
from tiberate.typing import Plaintext
from tiberate.utils.massive import (
    calculate_ckks_cipher_datastruct_size_in_list_recursive,
)

from .interface import (
    BenchmarkBase,
    BenchmarkResult,
    BenchmarkResultMetricType,
)

benchreg = Registry("benchmarks")


def test_lat_and_size_until_level_used_up(engine: CkksEngine) -> list[list]:
    result_table = []
    result_table.append(
        [
            'level (in->out)',
            'cc_add latency (ms)',
            'cc_mult latency (ms) (no relin)',
            'cc_add_triplet latency (ms)',
            'relin latency (ms)',
            'pt_add latency (ms)',
            'pt_mult latency (ms) (no rescale)',
            'rescale latency (ms)',
            'rotate latency (ms) (no key switching)',
            'key switching latency (ms)',
            'ct size (MB)',
            'pt cache size (MB)',
        ]
    )  # append header
    level_list = [0, *list(range(engine.num_levels))]  # first run is warmup
    try:
        for idx, i in enumerate(level_list):
            # create ct
            ct = engine.encodecrypt([1, 2, 3, 4], level=i)

            # ct size
            ct_size_mb = (
                calculate_ckks_cipher_datastruct_size_in_list_recursive(ct)
                / 1e6
            )

            # rotate
            time0 = time.time()
            ct_ = engine.rotate_single(
                ct, engine.rotk[1], post_key_switching=False
            )
            lat_rotate = time.time() - time0
            lat_rotate = lat_rotate * 1e3  # convert to ms

            # key switching
            time0 = time.time()
            ct_ = engine.switch_key(ct_, ksk=engine.rotk[1])
            lat_key_switch = time.time() - time0
            lat_key_switch = lat_key_switch * 1e3  # convert to ms

            # cadd
            time0 = time.time()
            ct_ = engine.cc_add(ct, ct)
            lat_cadd = time.time() - time0
            lat_cadd = lat_cadd * 1e3  # convert to ms

            # cmult no relin
            time0 = time.time()
            ct_ = engine.cc_mult(ct, ct, post_relin=False)
            lat_cmult_no_relin = time.time() - time0
            lat_cmult_no_relin = lat_cmult_no_relin * 1e3  # convert to ms

            # cadd_tri
            time0 = time.time()
            ct_ = engine.cc_add_triplet(ct_, ct_)
            lat_cadd_tri = time.time() - time0
            lat_cadd_tri = lat_cadd_tri * 1e3  # convert to ms

            # relin
            time0 = time.time()
            ct_ = engine.relinearize(ct_)
            lat_relin = time.time() - time0
            lat_relin = lat_relin * 1e3  # convert to ms

            # create pt
            pt = Plaintext([1, 2, 3, 4])

            # pcadd
            ct_ = engine.pc_add(pt=pt, ct=ct)
            time0 = time.time()
            ct_ = engine.pc_add(pt=pt, ct=ct)
            lat_pcadd = time.time() - time0
            lat_pcadd = lat_pcadd * 1e3  # convert to ms

            # pt add cache size
            pt_add_cache_size = (
                calculate_ckks_cipher_datastruct_size_in_list_recursive(pt.data)
                / 1e6
            )
            pt = Plaintext([1, 2, 3, 4])  # reset pt

            # pcmult no rescale
            ct_ = engine.pc_mult(pt=pt, ct=ct, post_rescale=False)
            time0 = time.time()
            ct_ = engine.pc_mult(pt=pt, ct=ct, post_rescale=False)
            lat_pcmult_no_rescale = time.time() - time0
            lat_pcmult_no_rescale = lat_pcmult_no_rescale * 1e3  # convert to ms

            # rescale
            time0 = time.time()
            ct_ = engine.rescale(ct_)
            lat_rescale = time.time() - time0
            lat_rescale = lat_rescale * 1e3  # convert to ms

            # pt mult cache size
            pt_mult_cache_size = (
                calculate_ckks_cipher_datastruct_size_in_list_recursive(pt.data)
                / 1e6
            )

            # print result, do not drop precision
            result_table.append(
                [
                    f"{ct.level}->{ct_.level}" if idx else "warmup",
                    lat_cadd,
                    lat_cmult_no_relin,
                    lat_cadd_tri,
                    lat_relin,
                    lat_pcadd,
                    lat_pcmult_no_rescale,
                    lat_rescale,
                    lat_rotate,
                    lat_key_switch,
                    ct_size_mb,
                    pt_add_cache_size,
                    # pt_mult_cache_size,
                ]
            )
    except errors.MaximumLevelError as e:
        pass  # reached max level, this is expected
    except Exception as e:
        raise e

    return result_table


@benchreg.register(name="Test op latency at each level")
class ConsumeAllLevelsBenchmark(BenchmarkBase):
    def __init__(self):
        self.name = "Test op latency at each level"
        self.description = "Evaluate the performance of all operations at each level, including ciphertext addition, multiplication, plaintext addition, multiplication, rotation, and key switching, as well as ciphertext size and plaintext cache size."
        self.config_matrix = {
            "logN14": {
                "description": "Using polynomial degree logN14",
                "ckks_params": Preset.logN14,
                "relinearize": True,
            },
            "logN15": {
                "description": "Using polynomial degree logN15",
                "ckks_params": Preset.logN15,
                "relinearize": True,
            },
            "logN16": {
                "description": "Using polynomial degree logN16",
                "ckks_params": Preset.logN16,
                "relinearize": True,
            },
        }

    def get_option_name2desc(self):
        return {k: v["description"] for k, v in self.config_matrix.items()}

    def run(self, option_name):
        assert (
            option_name in self.config_matrix
        ), f"Invalid benchmark name: {option_name}"
        config = self.config_matrix[option_name]
        ckks_params = config["ckks_params"]
        logger.info(f"Running benchmark: {config['description']}")

        benchmark_result = BenchmarkResult()

        engine = CkksEngine(ckks_params)

        # =========== Test Error ========== #
        packed_ct_1, input_tensor_1 = engine.randn(return_src=True)
        _, input_tensor_2 = engine.randn(return_src=True)

        try:
            max_diff_array = []
            mean_diff_array = []
            while True:
                # At current level check the error
                dec_he_out = engine.decryptcode(packed_ct_1)
                diff = input_tensor_1 - dec_he_out
                max_diff = diff.float().max()
                mean_diff = diff.float().mean()
                max_diff_array.append(max_diff)
                mean_diff_array.append(mean_diff)

                # Ecrypt the input tensor to ciphertext
                packed_ct_2 = engine.encodecrypt(
                    input_tensor_2, level=packed_ct_1.level
                )

                # Perform the CMult operation to increase the level
                packed_ct_1 = engine.cc_mult(packed_ct_1, packed_ct_2)
                input_tensor_1 = input_tensor_1 * input_tensor_2

                # Perform CAdd+1 then CSub-1
                packed_ct_1 = engine.cc_add(packed_ct_1, packed_ct_2)
                packed_ct_1 = engine.cc_sub(packed_ct_1, packed_ct_2)

                # Do rotate
                packed_ct_1 = engine.rotate_single(
                    ct=packed_ct_1, rotk=engine.rotk[1]
                )
                input_tensor_1 = torch.roll(input_tensor_1, shifts=1, dims=0)

                packed_ct_1 = engine.pc_add(
                    pt=Plaintext([1]),
                    ct=packed_ct_1,
                )
                input_tensor_1[0] += 1

                packed_ct_1 = engine.pc_mult(pt=Plaintext([1]), ct=packed_ct_1)
                input_tensor_1[0] *= 1

        except errors.MaximumLevelError as e:
            pass  # reached max level
        except Exception as e:
            raise e  # unexpected error
        finally:
            benchmark_result.add_metric(
                name="Max diff",
                metric_type=BenchmarkResultMetricType.PLOT,
                value=max_diff_array,
                series="error",
                description="the maximum difference between the expected output and decrypted output at each level.",
            )
            benchmark_result.add_metric(
                name="Mean diff",
                metric_type=BenchmarkResultMetricType.PLOT,
                value=mean_diff_array,
                series="error",
                description="the mean difference between the expected output and decrypted output at each level.",
            )
            # benchmark_result.add_metric(
            #     name="Last level error distribution",
            #     metric_type=BenchmarkResultMetricType.DISTRIBUTION,
            #     series="error",
            #     value=diff.real.tolist(),
            #     description="The distribution of the last level error.",
            # )

        # =========== Test Latency and Size ========== #

        lat_table = test_lat_and_size_until_level_used_up(engine)
        benchmark_result.add_metric(
            name="Latency / Data Size by Level",
            metric_type=BenchmarkResultMetricType.TABLE,
            value=lat_table,
            description="Latency and size/level",
        )

        """Prompt the user to optionally save benchmark results to a file."""
        if click.confirm(
            "Do you want to save the result after benchmark?", default=False
        ):
            default_filename = f"./{self.name}_{option_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            default_filename = legal_file_name_of(default_filename)
            file_name = click.prompt(
                "Enter file name to save the benchmark result",
                default=default_filename,
                show_default=True,
            )
            try:
                with open(file_name, "w") as f:
                    for row in lat_table:
                        f.write(",".join(map(str, row)) + "\n")
                logger.info(f"Benchmark result saved to {file_name}.")
            except Exception as e:
                logger.error(f"Failed to save benchmark result: {e}")
        else:
            logger.info("Benchmark result not saved.")

        return benchmark_result


if __name__ == "__main__":
    benchmark = ConsumeAllLevelsBenchmark()
    benchmark.run("logN14")  # Example run with logN14
