import math
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from tiberate import errors
from tiberate.config.security_parameters import maximum_qbits
from tiberate.utils.generate_primes import (
    generate_message_primes,
    generate_scale_primes,
)


class Preset(Enum):
    logN14 = "logN14"
    logN15 = "logN15"
    logN16 = "logN16"
    logN17 = "logN17"


_PRESET_CONFIGS = {
    Preset.logN14: {
        "logN": 14,
        "num_special_primes": 1,
    },
    Preset.logN15: {
        "logN": 15,
        "num_special_primes": 2,
    },
    Preset.logN16: {
        "logN": 16,
        "num_special_primes": 4,
    },
    Preset.logN17: {
        "logN": 17,
        "num_special_primes": 6,
    },
}


@dataclass
class CkksConfig:
    buffer_bit_length: int = 62
    scale_bits: int = 40
    logN: int = 15
    num_scales: int | None = None
    num_special_primes: int = 2
    sigma: float = 3.2
    uniform_ternary_secret: bool = True
    security_bits: int = 128
    quantum: str = "post_quantum"
    distribution: str = "uniform"
    force_secured: bool = True

    @classmethod
    def from_preset(cls, preset: Preset, **kwargs):
        """
        Create a CkksConfig instance from a preset.
        Args:
            preset (Preset): The preset to use.
            **kwargs: Additional keyword arguments to override the preset values.
        Returns:
            CkksConfig: The CkksConfig instance with the preset values.
        """
        preset_config = _PRESET_CONFIGS[preset]
        instance = cls(**preset_config, **kwargs)
        return instance

    def __post_init__(self):

        self.N = 2**self.logN  # Polynomial length.

        self.int_scale = 2**self.scale_bits
        self.scale = np.float64(self.int_scale)

        # We set the message prime to of bit-length W-2.
        self.message_bits = self.buffer_bit_length - 2

        if self.uniform_ternary_secret:
            self.secret_key_sampling_method = "uniform ternary"
        else:
            self.secret_key_sampling_method = "sparse ternary"

        # dtypes.
        self.torch_dtype = {30: torch.int32, 62: torch.int64}[
            self.buffer_bit_length
        ]
        self.numpy_dtype = {30: np.int32, 62: np.int64}[self.buffer_bit_length]

        # Read in pre-calculated high-quality primes.
        try:
            message_special_primes = generate_message_primes()[
                self.message_bits
            ][self.N]
        except KeyError:
            raise errors.NotFoundMessageSpecialPrimes(
                message_bit=self.message_bits, N=self.N
            )

        # For logN > 16, we need significantly more primes.
        how_many = 64 if self.logN < 16 else 128
        try:
            scale_primes = generate_scale_primes(how_many=how_many)[
                self.scale_bits, self.N
            ]
        except KeyError:
            raise errors.NotFoundScalePrimes(
                scale_bits=self.scale_bits, N=self.N
            )

        # Compose the primes pack.
        # Rescaling drops off primes in --> direction.
        # Key switching drops off primes in <-- direction.
        # Hence, [scale primes, base message prime, special primes]
        self.max_qbits = int(
            maximum_qbits(
                self.N, self.security_bits, self.quantum, self.distribution
            )
        )
        base_special_primes = message_special_primes[
            : 1 + self.num_special_primes
        ]

        # If num_scales is None, generate the maximal number of levels.
        try:
            if self.num_scales is None:
                base_special_bits = sum(
                    [math.log2(p) for p in base_special_primes]
                )
                available_bits = self.max_qbits - base_special_bits
                num_scales = 0
                available_bits -= math.log2(scale_primes[num_scales])
                while available_bits > 0:
                    num_scales += 1
                    available_bits -= math.log2(scale_primes[num_scales])

            self.num_scales = num_scales
            self.q = scale_primes[:num_scales] + base_special_primes
        except IndexError:
            raise errors.NotEnoughPrimes(scale_bits=self.scale_bits, N=self.N)

        # Check if security requirements are met.
        self.total_qbits = math.ceil(sum([math.log2(qi) for qi in self.q]))

        if self.total_qbits > self.max_qbits:
            if self.force_secured:
                raise errors.ViolatedAllowedQbits(
                    scale_bits=self.scale_bits,
                    N=self.N,
                    num_scales=self.num_scales,
                    max_qbits=self.max_qbits,
                    total_qbits=self.total_qbits,
                )
            else:
                warnings.warn(
                    f"Maximum allowed qbits are violated: "
                    f"max_qbits={self.max_qbits:4d} and the "
                    f"requested total is {self.total_qbits:4d}."
                )

    @property
    def generation_string(self):
        return (
            f"{self.buffer_bit_length}_{self.scale_bits}_{self.logN}_{self.num_scales}_"
            f"{self.num_special_primes}_{self.security_bits}_{self.quantum}_"
            f"{self.distribution}"
        )
