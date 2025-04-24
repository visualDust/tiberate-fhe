#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust

import math
from collections import deque

import numpy as np
import torch
from loguru import logger

from tiberate.typing import *


def copy_some_datastruct(src):
    if isinstance(src, DataStruct):
        return src.clone()
    if isinstance(src, torch.Tensor):
        return src.clone()
    if isinstance(src, np.ndarray):
        return src.copy()
    if isinstance(src, (list, tuple)):
        return [copy_some_datastruct(d) for d in src]
    if isinstance(src, dict):
        return {k: copy_some_datastruct(v) for k, v in src.items()}
    else:
        logger.warning(
            f"Unknown type: {type(src)} on copy. Will return the original."
        )
        return src


def calculate_tensor_size_in_bytes(tensor: torch.Tensor):
    shape = tensor.shape
    element_size = tensor.element_size()
    total_size = element_size
    for dim in shape:
        total_size *= dim
    return total_size


def calculate_ckks_cipher_datastruct_size_in_list_recursive(
    list_or_cipher: tuple | list | torch.Tensor | DataStruct,
):
    if isinstance(list_or_cipher, DataStruct):
        return calculate_ckks_cipher_datastruct_size_in_list_recursive(
            list_or_cipher.data
        )
    elif isinstance(list_or_cipher, (list, tuple)):
        return sum(
            calculate_ckks_cipher_datastruct_size_in_list_recursive(d)
            for d in list_or_cipher
        )
    elif isinstance(list_or_cipher, torch.Tensor):
        return calculate_tensor_size_in_bytes(list_or_cipher)
    elif isinstance(list_or_cipher, dict):
        return sum(
            calculate_ckks_cipher_datastruct_size_in_list_recursive(v)
            for v in list_or_cipher.values()
        )
    else:
        raise ValueError(f"Unknown type: {type(list_or_cipher)}")


def next_power_of_n(x: int, n: int):
    return n ** math.ceil(math.log(x, n))


def next_power_of_2(n: int):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def next_multiple_of_n(x: int, n: int):
    return n * math.ceil(x / n)


def decompose_with_power_of_2(a: int, n: int) -> list:
    """
    Decomposes an offset `a` into a sum of unit offsets that are powers of 2,
    assuming a modular space of size `n` where `n` is a power of 2.
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    if a < 0:
        a = n + a  # Convert negative offset to positive equivalent
    result = []
    expo = 0
    while (1 << expo) < n:
        unit = 1 << expo
        if a & unit:
            result.append(unit)
        expo += 1
    return result


def decompose_rot_offsets(offset: int, num_slots: int, rotks: dict) -> list:
    """
    Decompose a rotation offset into a list of smaller offsets that can be applied
    using available rotation keys (rotks) and powers of 2 up to num_slots // 2.

    Discards any decomposition that uses more steps than the ideal
    decompose_with_power_of_2 solution. The assumption is that it should use existing
    rotation keys first.
    """
    # Reference ideal decomposition
    best_possible_decomp = decompose_with_power_of_2(offset, num_slots)
    max_steps = len(best_possible_decomp)

    # Build list of available rotations
    available_rot_offsets = list(rotks.keys()) + [
        1 << i for i in range(int(math.log2(num_slots // 2)))
    ]
    available_rot_offsets = list(set(available_rot_offsets))
    available_rot_offsets.sort()

    # Limit the search space reasonably
    bound = num_slots

    visited = set()
    queue = deque()
    queue.append((0, []))
    visited.add(0)

    while queue:
        curr_sum, path = queue.popleft()
        if curr_sum == offset:
            if len(path) <= max_steps:
                return path
            else:
                break  # No better solution than power-of-2 version

        for coin in available_rot_offsets:
            next_sum = curr_sum + coin
            if -bound <= next_sum <= bound and next_sum not in visited:
                visited.add(next_sum)
                queue.append((next_sum, path + [coin]))

    # If no solution found, return solution given by decompose_with_power_of_2
    return best_possible_decomp


if __name__ == "__main__":
    print("Testing decompose_rot_offsets...\n")
    rotks = {  # Simulated already created rotation keys
        -1: None,
        -3: None,
        3: None,
        9: None,
        5: None,
    }
    num_slots = 16
    test_offsets = [3, 5, 7, -3, 15, 10, 0]
    for offset in test_offsets:
        print(f"Offset: {offset}")
        try:
            result = decompose_rot_offsets(offset, num_slots, rotks)
            po2 = decompose_with_power_of_2(offset, num_slots)
            print(f"  Power-of-2 decomposition: {po2}")
            print(f"  Decomposed with rotks:            {result}")
            assert len(result) <= len(
                po2
            ), "Decomposition with rotks is longer than power-of-2"
        except ValueError as e:
            print(f"  Error: {e}")
        print("-" * 50)
