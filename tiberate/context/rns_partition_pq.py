import functools
import operator
from dataclasses import dataclass

import numpy as np

from tiberate.config.ckks_config import CkksConfig


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
    num_devices: int = 1

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
    def from_ckks_config(cls, ckks_config: CkksConfig, num_devices: int = 1):
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
            num_devices=num_devices,
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
