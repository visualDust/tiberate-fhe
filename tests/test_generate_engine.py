import pytest

from tiberate import CkksEngine


@pytest.fixture()
def ckks_engine(
    devices: list[str] | None = None,
    logN: int = 15,
    scale_bits: int = 40,
    read_cache: bool = True,
):
    """
        generate ckks_engine
    @param devices:
    @param logN:
    @param scale_bits:
    @param read_cache:
    @return:
    """
    ctx_params = {
        "logN": logN,
        "scale_bits": scale_bits,
        "security_bits": 128,
        "num_scales": None,
        "num_special_primes": 2,
        "buffer_bit_length": 62,
        "sigma": 3.2,
        "uniform_ternary_secret": True,
        "cache_folder": "cache/",
        "quantum": "post_quantum",
        "distribution": "uniform",
        "read_cache": read_cache,
        "save_cache": True,
    }
    engine = CkksEngine(devices=devices, ckks_params=ctx_params)
    return engine


scales = list(range(20, 50, 5))
logNs = list(range(14, 17))
test_cases = [
    (logN, scale, True)
    for logN in list(range(14, 17))  # logNs
    for scale in list(range(20, 50, 5))  # scales
]


@pytest.mark.parametrize("ckks_engine", test_cases, indirect=["ckks_engine"])
def test_make_engine(ckks_engine):
    """
        test for generate ckks engine
    @param ckks_engine:
    @return:
    """
    assert isinstance(ckks_engine, CkksEngine)
