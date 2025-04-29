import functools
import math
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import torch

from tiberate import errors
from tiberate.cache import CACHE_FOLDER
from tiberate.prim.generate_primes import (
    generate_message_primes,
    generate_scale_primes,
)
from tiberate.security_parameters import maximum_qbits


class RngType(Enum):
    SIMPLE = "SimpleRNG"
    CSPRNG = "Csprng"


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
    logN: int = 16
    buffer_bit_length: int = 62
    scale_bits: int = 40
    num_special_primes: int = 2
    sigma = 3.2
    uniform_ternary_secret: bool = True
    security_bits: int = 128
    quantum: str = "post_quantum"
    distribution: str = "uniform"
    read_cache: bool = True
    save_cache: bool = True
    allow_sk_gen: bool = True
    bias_guard: bool = True
    norm: str = "forward"
    rng_class: RngType = RngType.CSPRNG
    cache_folder: str = CACHE_FOLDER
    num_scales: int | None = None

    # Derived attributes
    runtime_config: Union["CkksEngineRuntimeConfig", None] = None
    rns_partition: Union["RnsPartition", None] = None

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
        config = cls(**preset_config, **kwargs)
        return config

    def __post_init__(self):
        """
        Post-initialization processing for CkksConfig.
        """
        self.runtime_config = CkksEngineRuntimeConfig.from_ckks_config(self)
        self.rns_partition = RnsPartition.from_ckks_config(self)


@dataclass
class CkksEngineRuntimeConfig:
    # Derived attributes
    devices: list[str] | None
    N: int  # Polynomial length.
    max_qbits: int  # Maximum number of bits in the primes pack.
    num_slots: int  # Number of slots in the polynomial.
    num_levels: int  # Number of levels in the CKKS scheme.
    int_scale: int  # Integer scale for the CKKS scheme.
    scale: float  # Scale factor for the CKKS scheme.
    message_bits: int  # Bit-length of the message prime.
    q: list[int]  # List of primes in the CKKS scheme.
    R: int  # Bit-length of the CKKS scheme.
    torch_dtype: torch.dtype  # Data type for PyTorch tensors.
    numpy_dtype: np.dtype  # Data type for NumPy arrays.
    num_devices: int  # Number of devices available for computation.
    generation_string: str  # String representation of the configuration.
    R_square: list[int]  # List of R^2 mod q_i for each prime q_i.

    @classmethod
    def from_ckks_config(cls, ckks_config: CkksConfig):
        """
        Create a CkksEngineRuntimeConfig instance from a CkksConfig instance.
        Args:
            ckks_config (CkksConfig): The CkksConfig instance to use.
        Returns:
            CkksEngineRuntimeConfig: The CkksEngineRuntimeConfig instance.
        """

        devices = ["cuda:0"]  # always use cuda:0
        num_devices = len(devices)

        N = 2**ckks_config.logN

        # Compose the primes pack.
        # Rescaling drops off primes in --> direction.
        # Key switching drops off primes in <-- direction.
        # Hence, [scale primes, base message prime, special primes]
        max_qbits = int(
            maximum_qbits(
                N,
                ckks_config.security_bits,
                ckks_config.quantum,
                ckks_config.distribution,
            )
        )
        num_slots = N // 2

        int_scale = 2**ckks_config.scale_bits

        scale = np.float64(
            2**ckks_config.scale_bits
        )  # TODO(puqing): What is difference between int_scale and scale?

        # We set the message prime to of bit-length W-2.
        message_bits = ckks_config.buffer_bit_length - 2

        # Read in pre-calculated high-quality primes.
        try:
            message_special_primes = generate_message_primes(
                cache_folder=ckks_config.cache_folder
            )[message_bits][N]
        except KeyError:
            raise errors.NotFoundMessageSpecialPrimes(
                message_bit=message_bits, N=N
            )

        # For logN > 16, we need significantly more primes.
        how_many = 64 if ckks_config.logN < 16 else 128
        try:
            scale_primes = generate_scale_primes(
                cache_folder=ckks_config.cache_folder, how_many=how_many
            )[ckks_config.scale_bits, N]
        except KeyError:
            raise errors.NotFoundScalePrimes(
                scale_bits=ckks_config.scale_bits, N=N
            )

        base_special_primes = message_special_primes[
            : 1 + ckks_config.num_special_primes
        ]

        # If num_scales is None, generate the maximal number of levels.
        try:
            if ckks_config.num_scales is None:
                base_special_bits = sum(
                    [math.log2(p) for p in base_special_primes]
                )
                available_bits = max_qbits - base_special_bits
                num_scales = 0
                available_bits -= math.log2(scale_primes[num_scales])
                while available_bits > 0:
                    num_scales += 1
                    available_bits -= math.log2(scale_primes[num_scales])

            ckks_config.num_scales = num_scales
            q = scale_primes[:num_scales] + base_special_primes

        except IndexError:
            raise errors.NotEnoughPrimes(scale_bits=ckks_config.scale_bits, N=N)

        num_levels = (
            ckks_config.num_scales
        )  # TODO(puqing): Why not use ckks_config.num_scales?

        R = 2**ckks_config.buffer_bit_length

        torch_dtype = {30: torch.int32, 62: torch.int64}[
            ckks_config.buffer_bit_length
        ]

        generation_string = (
            f"{ckks_config.buffer_bit_length}_{ckks_config.scale_bits}_{ckks_config.logN}_{ckks_config.num_scales}_"
            f"{ckks_config.num_special_primes}_{ckks_config.security_bits}_{ckks_config.quantum}_"
            f"{ckks_config.distribution}"
        )

        R_square = [R**2 % qi for qi in q]

        numpy_dtype = {30: np.int32, 62: np.int64}[
            ckks_config.buffer_bit_length
        ]

        return cls(
            devices=devices,
            N=N,
            max_qbits=max_qbits,
            num_slots=num_slots,
            num_levels=num_levels,
            int_scale=int_scale,
            scale=scale,
            message_bits=message_bits,
            q=q,
            R=R,
            torch_dtype=torch_dtype,
            numpy_dtype=numpy_dtype,
            num_devices=num_devices,
            generation_string=generation_string,
            R_square=R_square,
        )


@dataclass
class RnsPartition:
    """
    Manages the partitioning of the RNS primes for a given number of devices.

    Attributes:
        num_ordinary_primes (int): Number of ordinary primes.
        num_special_primes (int): Number of special primes.
        num_devices (int): Number of devices.

        partitions (list[list[int]]): List of partitions for the ordinary primes, The last partition contains the
            special primes.
        part_allocations (list[list[int]]): List of allocations for the primes.
        flat_prime_allocations (list[list[int]]): Flattened list of prime allocations for each device.

    Example:
        >>> rns_partition = RnsPartition(num_ordinary_primes=4, num_special_primes=2, num_devices=2)
        >>> print(rns_partition.partitions)
        [[0, 1], [2], [3], [4, 5]]
        >>> print(rns_partition.part_allocations)
        [[1, 2, 3], [0, 3]]
        >>> print(rns_partition.flat_prime_allocations)
        [[2, 3, 4, 5], [0, 1, 4, 5]]
        >>> print(rns_partition.destination_arrays)
        [[[2, 3], [0, 1]], [[2, 3], [1]], [[2, 3]], [[3]]]
        >>> print(rns_partition.destination_arrays_with_special)
        [[[2, 3, 4, 5], [0, 1, 4, 5]],
         [[2, 3, 4, 5], [1, 4, 5]],
         [[2, 3, 4, 5], [4, 5]],
         [[3, 4, 5], [4, 5]]]
        >>> print(rns_partition.d)
        [[2, 3], [0, 1]]
    """

    # Configuration parameters
    num_ordinary_primes: int = 17
    num_special_primes: int = 2
    num_devices: int = 2

    # Derived attributes
    base_prime_idx = num_ordinary_primes - 1
    partitions: list[list[int]] | None = None
    part_allocations: list[list[int]] | None = None
    flat_prime_allocations: list[list[int]] | None = None
    destination_arrays: list[list[list[int]]] | None = None
    destination_arrays_with_special: list[list[list[int]]] | None = None
    prime_allocations: list[list[int]] | None = None
    d: list[list[int]] | None = None
    rescaler_loc: list[int] | None = None

    @classmethod
    def from_ckks_config(cls, ckks_config: CkksConfig):
        """
        Create a RnsPartition instance from a CkksConfig instance.
        Args:
            ckks_config (CkksConfig): The CkksConfig instance to use.
        Returns:
            RnsPartition: The RnsPartition instance.
        """
        self = cls(
            num_ordinary_primes=ckks_config.num_scales + 1,
            num_special_primes=ckks_config.num_special_primes,
            num_devices=len(ckks_config.runtime_config.devices),
        )
        return self

    def __post_init__(self):
        primes_idx = list(range(self.num_ordinary_primes - 1))

        num_partitions = -(
            -(self.num_ordinary_primes - 1) // self.num_special_primes
        )

        part = lambda i: primes_idx[
            i * self.num_special_primes : (i + 1) * self.num_special_primes
        ]
        partitions = [part(i) for i in range(num_partitions)]
        partitions.append([self.num_ordinary_primes - 1])
        partitions.append(
            list(
                range(
                    self.num_ordinary_primes,
                    self.num_ordinary_primes + self.num_special_primes,
                )
            )
        )
        self.partitions = partitions

        alloc = lambda i: list(
            range(num_partitions - i - 1, -1, -self.num_devices)
        )[::-1]
        part_allocations = [alloc(i) for i in range(self.num_devices)]
        part_allocations[0].append(num_partitions)

        self.d = [
            [p for part in part_allocations[i] for p in partitions[part]]
            for i in range(self.num_devices)
        ]

        for p in part_allocations:
            p.append(num_partitions + 1)
        self.part_allocations = part_allocations

        expand_alloc = lambda i: [
            partitions[part] for part in part_allocations[i]
        ]
        prime_allocations = [expand_alloc(i) for i in range(self.num_devices)]
        self.prime_allocations = prime_allocations

        self.flat_prime_allocations = [
            functools.reduce(operator.iadd, alloc, [])
            for alloc in prime_allocations
        ]

        filter_alloc = lambda devi, i: [
            a for a in self.flat_prime_allocations[devi] if a >= i
        ]
        self.destination_arrays_with_special = []
        for lvl in range(self.num_ordinary_primes):
            src = [filter_alloc(devi, lvl) for devi in range(self.num_devices)]
            self.destination_arrays_with_special.append(src)

        special_removed = lambda i: [
            a[: -self.num_special_primes]
            for a in self.destination_arrays_with_special[i]
        ]
        self.destination_arrays = [
            special_removed(i) for i in range(self.num_ordinary_primes)
        ]
        lint = lambda arr: [a for a in arr if len(a) > 0]
        self.destination_arrays = [lint(a) for a in self.destination_arrays]

        # -------------------------------
        # The shit of compute_partitions
        # -------------------------------
        self.part_cumsums = []
        self.part_counts = []
        self.parts = []
        self.destination_parts = []
        self.destination_parts_with_special = []
        self.p = []
        self.p_special = []
        self.diff = []

        self.d_special = [
            self.destination_arrays_with_special[0][dev_i]
            for dev_i in range(self.num_devices)
        ]

        for lvl in range(self.num_ordinary_primes):
            pcu, pco, par = self.partings(lvl)
            self.part_cumsums.append(pcu)
            self.part_counts.append(pco)
            self.parts.append(par)

            dest = self.destination_arrays_with_special[lvl]
            destp_special = [
                [[d[pi] for pi in p] for p in dev_p]
                for d, dev_p in zip(dest, par)
            ]
            destp = [dev_dp[:-1] for dev_dp in destp_special]

            self.destination_parts.append(destp)
            self.destination_parts_with_special.append(destp_special)

            diff = [
                len(d1) - len(d2)
                for d1, d2 in zip(
                    self.destination_arrays_with_special[0],
                    self.destination_arrays_with_special[lvl],
                )
            ]
            p_special = [
                [[pi + d for pi in p] for p in dev_p]
                for d, dev_p in zip(diff, self.parts[lvl])
            ]
            p = [dev_p[:-1] for dev_p in p_special]

            self.p.append(p)
            self.p_special.append(p_special)
            self.diff.append(diff)

        # -------------------------------
        # The shit of compute_rescaler_locations
        # -------------------------------
        mins = lambda arr: [min(a) for a in arr]
        mins_loc = lambda a: mins(a).index(min(mins(a)))
        self.rescaler_loc = [
            mins_loc(a) for a in self.destination_arrays_with_special
        ]

    def partings(self, lvl):
        count_element_sizes = lambda arr: np.array([len(a) for a in arr])
        cumsum_element_sizes = lambda arr: np.cumsum(arr)
        remove_empty_parts = lambda arr: [a for a in arr if a > 0]
        regenerate_parts = lambda arr: [
            list(range(a, b)) for a, b in zip([0] + arr[:-1], arr)
        ]

        part_counts = [count_element_sizes(a) for a in self.prime_allocations]
        part_cumsums = [cumsum_element_sizes(a) for a in part_counts]
        level_diffs = [
            len(a) - len(b)
            for a, b in zip(
                self.destination_arrays_with_special[0],
                self.destination_arrays_with_special[lvl],
            )
        ]

        part_cumsums_lvl = [
            remove_empty_parts(a - d) for a, d in zip(part_cumsums, level_diffs)
        ]
        part_count_lvl = [np.diff(a, prepend=0) for a in part_cumsums_lvl]
        parts_lvl = [regenerate_parts(a) for a in part_cumsums_lvl]
        return part_cumsums_lvl, part_count_lvl, parts_lvl
