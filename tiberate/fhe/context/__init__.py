from .ckks_context import CkksContext
from .security_parameters import maximum_qbits, minimum_cyclotomic_order

presets = {
    "bronze": {
        "logN": 14,
        "num_special_primes": 1,
        "scale_bits": 40,
        "num_scales": None,
    },
    "silver": {
        "logN": 15,
        "num_special_primes": 2,
        "scale_bits": 40,
        "num_scales": None,
    },
    "gold": {
        "logN": 16,
        "num_special_primes": 4,
        "scale_bits": 40,
        "num_scales": None,
    },
    "platinum": {
        "logN": 17,
        "num_special_primes": 6,
        "scale_bits": 40,
        "num_scales": None,
    },
}
