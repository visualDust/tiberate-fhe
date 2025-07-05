import torch

# =============================================================
# Montgomery Arithmetic Operations
# Wrappers for mont.cpp and mont_fused.cpp
# =============================================================


def mont_mult(
    a: torch.Tensor,
    b: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> torch.Tensor:
    """Perform Montgomery multiplication on two tensors.
    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        ql (torch.Tensor): Lower part of the modulus.
        qh (torch.Tensor): Upper part of the modulus.
        kl (torch.Tensor): Lower part of the Montgomery constant.
        kh (torch.Tensor): Upper part of the Montgomery constant.
    Returns:
        torch.Tensor: Result of Montgomery multiplication.
    """
    return torch.ops.tiberate_ntt_ops.mont_mult(a, b, ql, qh, kl, kh)


def mont_enter(
    a: torch.Tensor,
    Rs: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Prepare the tensor for Montgomery multiplication.
    Args:
        a (torch.Tensor): Tensor to prepare.
        Rs (torch.Tensor): Montgomery constant.
        ql (torch.Tensor): Lower part of the modulus.
        qh (torch.Tensor): Upper part of the modulus.
        kl (torch.Tensor): Lower part of the Montgomery constant.
        kh (torch.Tensor): Upper part of the Montgomery constant.
    """
    return torch.ops.tiberate_ntt_ops.mont_enter(a, Rs, ql, qh, kl, kh)


def mont_reduce(
    a: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Reduce the tensor after Montgomery multiplication.
    Args:
        a (torch.Tensor): Tensor to reduce.
        ql (torch.Tensor): Lower part of the modulus.
        qh (torch.Tensor): Upper part of the modulus.
        kl (torch.Tensor): Lower part of the Montgomery constant.
        kh (torch.Tensor): Upper part of the Montgomery constant.
    """
    return torch.ops.tiberate_ntt_ops.mont_reduce(a, ql, qh, kl, kh)


def mont_add(
    a: torch.Tensor,
    b: torch.Tensor,
    _2q: torch.Tensor,
) -> torch.Tensor:
    """Add two tensors in Montgomery form.
    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        _2q (torch.Tensor): Double of the modulus.
    Returns:
        torch.Tensor: Result of the addition.
    """
    return torch.ops.tiberate_ntt_ops.mont_add(a, b, _2q)


def mont_sub(
    a: torch.Tensor,
    b: torch.Tensor,
    _2q: torch.Tensor,
) -> torch.Tensor:
    """Subtract two tensors in Montgomery form.
    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        _2q (torch.Tensor): Double of the modulus.
    Returns:
        torch.Tensor: Result of the subtraction.
    """
    return torch.ops.tiberate_ntt_ops.mont_sub(a, b, _2q)


def reduce_2q(
    a: torch.Tensor,
    _2q: torch.Tensor,
) -> None:
    """Reduce the tensor to the range [0, 2q).
    Args:
        a (torch.Tensor): Tensor to reduce.
        _2q (torch.Tensor): Double of the modulus.
    """
    return torch.ops.tiberate_ntt_ops.reduce_2q(a, _2q)


def make_signed(
    a: torch.Tensor,
    _2q: torch.Tensor,
) -> None:
    """Convert the tensor to signed form.
    Args:
        a (torch.Tensor): Tensor to convert.
        _2q (torch.Tensor): Double of the modulus.
    """
    return torch.ops.tiberate_ntt_ops.make_signed(a, _2q)


def make_unsigned(
    a: torch.Tensor,
    _2q: torch.Tensor,
) -> None:
    """Convert the tensor to unsigned form.
    Args:
        a (torch.Tensor): Tensor to convert.
        _2q (torch.Tensor): Double of the modulus.
    """
    return torch.ops.tiberate_ntt_ops.make_unsigned(a, _2q)


def tile_unsigned(
    a: torch.Tensor,
    _2q: torch.Tensor,
) -> torch.Tensor:
    """Tile the unsigned tensor.
    Args:
        a (torch.Tensor): Tensor to tile.
        _2q (torch.Tensor): Double of the modulus.
    Returns:
        torch.Tensor: Tiled tensor.
    """
    return torch.ops.tiberate_ntt_ops.tile_unsigned(a, _2q)


def mont_add_many_3d(
    input: torch.Tensor,
    _2q: torch.Tensor,
) -> torch.Tensor:
    """Add many 3D tensors in Montgomery form.
    Args:
        input (torch.Tensor): List of tensors to add.
        _2q (torch.Tensor): Double of the modulus.
    Returns:
        torch.Tensor: Result of the addition.
    """
    return torch.ops.tiberate_fused_ops.mont_add_many_3d(input, _2q)


def mont_reduce_add_many_3d(
    input: torch.Tensor,
    _2q: torch.Tensor,
) -> torch.Tensor:
    """Reduce and add many 3D tensors in Montgomery form.
    Args:
        input (torch.Tensor): List of tensors to reduce and add.
        _2q (torch.Tensor): Double of the modulus.
    Returns:
        torch.Tensor: Result of the reduction and addition.
    """
    return torch.ops.tiberate_fused_ops.mont_reduce_add_many_3d(input, _2q)


def mont_add_reduce_2q(
    a: torch.Tensor,
    b: torch.Tensor,
    _2q: torch.Tensor,
) -> torch.Tensor:
    """Add two tensors in Montgomery form and reduce to 2q.
    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        _2q (torch.Tensor): Double of the modulus.
    Returns:
        torch.Tensor: Result of the addition and reduction.
    """
    return torch.ops.tiberate_fused_ops.mont_add_reduce_2q(a, b, _2q)


def mont_sub_reduce_2q(
    a: torch.Tensor,
    b: torch.Tensor,
    _2q: torch.Tensor,
) -> torch.Tensor:
    """Subtract two tensors in Montgomery form and reduce to 2q.
    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        _2q (torch.Tensor): Double of the modulus.
    Returns:
        torch.Tensor: Result of the subtraction and reduction.
    """
    return torch.ops.tiberate_fused_ops.mont_sub_reduce_2q(a, b, _2q)


def mont_enter_reduce_2q(
    a: torch.Tensor,
    Rs: torch.Tensor,
    _2q: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> torch.Tensor:
    """Prepare the tensor for Montgomery multiplication and reduce to 2q.
    Args:
        a (torch.Tensor): Tensor to prepare.
        Rs (torch.Tensor): Montgomery constant.
        _2q (torch.Tensor): Double of the modulus.
        ql (torch.Tensor): Lower part of the modulus.
        qh (torch.Tensor): Upper part of the modulus.
        kl (torch.Tensor): Lower part of the Montgomery constant.
        kh (torch.Tensor): Upper part of the Montgomery constant.
    Returns:
        torch.Tensor: Prepared tensor.
    """
    return torch.ops.tiberate_fused_ops.mont_enter_reduce_2q(
        a, Rs, _2q, ql, qh, kl, kh
    )


def rescale_exact_rounding_fused(
    a: torch.Tensor,
    Rs: torch.Tensor,
    rescaler: torch.Tensor,
    round_at: int,
    _2q: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Rescale the tensor with exact rounding.
    Args:
        a (torch.Tensor): Tensor to rescale.
        Rs (torch.Tensor): Montgomery constant.
        rescaler (torch.Tensor): Rescaling factor.
        round_at (int): Rounding position.
        _2q (torch.Tensor): Double of the modulus.
        ql (torch.Tensor): Lower part of the modulus.
        qh (torch.Tensor): Upper part of the modulus.
        kl (torch.Tensor): Lower part of the Montgomery constant.
        kh (torch.Tensor): Upper part of the Montgomery constant.
    """
    return torch.ops.tiberate_fused_ops.rescale_exact_rounding_fused(
        a, Rs, rescaler, round_at, _2q, ql, qh, kl, kh
    )


def rescale_non_exact_rounding_fused(
    a: torch.Tensor,
    Rs: torch.Tensor,
    rescaler: torch.Tensor,
    _2q: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Rescale the tensor with non-exact rounding.
    Args:
        a (torch.Tensor): Tensor to rescale.
        Rs (torch.Tensor): Montgomery constant.
        rescaler (torch.Tensor): Rescaling factor.
        _2q (torch.Tensor): Double of the modulus.
        ql (torch.Tensor): Lower part of the modulus.
        qh (torch.Tensor): Upper part of the modulus.
        kl (torch.Tensor): Lower part of the Montgomery constant.
        kh (torch.Tensor): Upper part of the Montgomery constant.
    """
    return torch.ops.tiberate_fused_ops.rescale_non_exact_rounding_fused(
        a, Rs, rescaler, _2q, ql, qh, kl, kh
    )
