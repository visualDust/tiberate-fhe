import time

import plotext as plt
from loguru import logger
from vdtoys.registry import Registry

from tiberate import CkksConfig, CkksEngine, Preset
from tiberate.typing import Plaintext
from tiberate.utils.massive import (
    calculate_ckks_cipher_datastruct_size_in_list_recursive,
)

from .interface import BenchmarkBase

benchreg = Registry("benchmarks")


def test_lat_and_size_until_level_used_up(
    engine: CkksEngine, do_print: bool = True
):
    # print table header
    if do_print:
        print(
            "ct level, ct size(MB),cc_add latency(ms),cc_mult latency(ms)(no relin),cc_add_triplet latency(ms),relin latency(ms),pt_add latency(ms),pt_add cache size(MB),pt_mult latency(ms)(no rescale),rescale latency(ms),pt_mult cache size(MB),rotate latency(ms)(no key switching),key switching latency(ms)"
        )

    try:
        for i in range(engine.num_levels - 1):
            # create ct
            ct = engine.encodecrypt([1, 2, 3, 4], level=i)

            # ct size
            size_mb = (
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
            if do_print:
                print(
                    f"{ct.level},{size_mb},{lat_cadd},{lat_cmult_no_relin},{lat_cadd_tri},{lat_relin},{lat_pcadd},{pt_add_cache_size},{lat_pcmult_no_rescale},{lat_rescale},{pt_mult_cache_size},{lat_rotate},{lat_key_switch}"
                )

    except Exception as e:
        raise e


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

        # =========== Test Error ========== #
        packed_ct_1, input_tensor_1 = engine.randn(return_src=True)
        _, input_tensor_2 = engine.randn(return_src=True)

        try:
            max_diff_array = []
            mean_diff_array = []
            while True:
                # Perform the CMult operation
                packed_ct_2 = engine.encodecrypt(
                    input_tensor_2, level=packed_ct_1.level
                )
                packed_ct_1 = engine.cc_mult(packed_ct_1, packed_ct_2)
                input_tensor_1 = input_tensor_1 * input_tensor_2
                dec_he_out = engine.decryptcode(packed_ct_1)

                # Check the difference
                diff = input_tensor_1 - dec_he_out
                max_diff = diff.max()
                mean_diff = diff.mean()
                max_diff_array.append(max_diff.real)
                mean_diff_array.append(mean_diff.real)

                logger.info(
                    f"At level {packed_ct_1.level}, Max diff: {max_diff}, Mean diff: {mean_diff}"
                )
        except Exception as e:
            logger.info(f"Seems max level reached: {e}")
            # raise e
        finally:
            plt.plot(max_diff_array, label="Max Diff")
            plt.plot(mean_diff_array, label="Mean Diff")
            plt.title("Max and Mean Error at Each Level")
            plt.show()

        # =========== Test Latency and Size ========== #

        test_lat_and_size_until_level_used_up(engine, do_print=False)  # warm up

        logger.info("<== Begin of CSV output ==>")
        test_lat_and_size_until_level_used_up(
            engine, do_print=True
        )  # print results
        logger.info("<== End of CSV output ==>")

        logger.info(
            f"Benchmark {self.name} with config {config['description']} completed successfully."
        )


if __name__ == "__main__":
    benchmark = ConsumeAllLevelsBenchmark()
    benchmark.run("logN14")  # Example run with logN14
