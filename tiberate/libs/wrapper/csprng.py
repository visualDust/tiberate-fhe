import torch


def chacha20(
    input: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Apply the ChaCha20 algorithm to the input tensor.

    Args:
        input (torch.Tensor): Input tensor to be processed.
        step (int): Step size for the ChaCha20 algorithm.

    Returns:
        torch.Tensor: Processed tensor after applying ChaCha20.
    """
    return torch.ops.tiberate_csprng_ops.chacha20(input, step)


def randround(
    input: torch.Tensor,
    rand_bytes: torch.Tensor,
) -> None:
    """Round the input tensor using random bytes.

    Args:
        input (torch.Tensor): Input tensor to be rounded.
        rand_bytes (torch.Tensor): Random bytes used for rounding.
    """
    return torch.ops.tiberate_csprng_ops.randround(input, rand_bytes)


def randint(
    input: torch.Tensor,
    q_ptrs: torch.Tensor,
) -> None:
    """Generate random integers in the input tensor.

    Args:
        input (torch.Tensor): Input tensor to be filled with random integers.
        q_ptrs (torch.Tensor): Pointers to the modulus values for random generation.
    """
    return torch.ops.tiberate_csprng_ops.randint(input, q_ptrs)


def randint_fast(
    input: torch.Tensor,
    q_ptrs: torch.Tensor,
    shift: int,
    step: int,
) -> torch.Tensor:
    """Generate random integers in the input tensor with fast method.

    Args:
        input (torch.Tensor): Input tensor to be filled with random integers.
        q_ptrs (torch.Tensor): Pointers to the modulus values for random generation.
        shift (int): Shift value for the random integers.
        step (int): Step value for the random integers.
    Returns:
        torch.Tensor: Tensor containing the generated random integers.
    """
    return torch.ops.tiberate_csprng_ops.randint_fast(
        input, q_ptrs, shift, step
    )


def discrete_gaussian(
    input: torch.Tensor,
    btree_ptr: int,
    btree_size: int,
    depth: int,
) -> None:
    """Apply discrete Gaussian noise to the input tensor.

    Args:
        input (torch.Tensor): Input tensor to be modified.
        btree_ptr (int): Pointer to the binary tree structure.
        btree_size (int): Size of the binary tree.
        depth (int): Depth of the binary tree.
    """
    return torch.ops.tiberate_csprng_ops.discrete_gaussian(
        input, btree_ptr, btree_size, depth
    )


def discrete_gaussian_fast(
    input: torch.Tensor,
    btree_ptr: int,
    btree_size: int,
    depth: int,
    step: int,
) -> torch.Tensor:
    """Apply fast discrete Gaussian noise to the input tensor.

    Args:
        input (torch.Tensor): Input tensor to be modified.
        btree_ptr (int): Pointer to the binary tree structure.
        btree_size (int): Size of the binary tree.
        depth (int): Depth of the binary tree.
        step (int): Step size for fast processing.

    Returns:
        torch.Tensor: Processed tensor after applying fast discrete Gaussian noise.
    """
    return torch.ops.tiberate_csprng_ops.discrete_gaussian_fast(
        input, btree_ptr, btree_size, depth, step
    )
