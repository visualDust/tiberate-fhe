_PRESETS = {
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

logN14 = _PRESETS["bronze"]
logN15 = _PRESETS["silver"]
logN16 = _PRESETS["gold"]
logN17 = _PRESETS["platinum"]

logN2preset = {
    14: logN14,
    15: logN15,
    16: logN16,
    17: logN17,
}
