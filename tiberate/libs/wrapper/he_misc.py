import torch


def pc_add_fused(
    ct_data: torch.Tensor,
    pt_data: torch.Tensor,
    Rs: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
    _2q: torch.Tensor,
) -> torch.Tensor:
    """Perform polynomial addition with ciphertext and plaintext data."""
    return torch.ops.tiberate_he_ops.pc_add_fused(
        ct_data, pt_data, Rs, ql, qh, kl, kh, _2q
    )


def switch_key_switch_later_part_extend(
    rns_len: int,
    state: torch.Tensor,
    l_enter: torch.Tensor,
    l_enter_start_offset: int,
    _2q: torch.Tensor,
    Rs: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> torch.Tensor:
    """Extend the key switch later part."""
    return torch.ops.tiberate_he_ops.switch_key_switch_later_part_extend(
        rns_len, state, l_enter, l_enter_start_offset, _2q, Rs, ql, qh, kl, kh
    )


def codec_rotate_make_unsigned_reduce_2q(
    a: torch.Tensor,
    perm: torch.Tensor,
    _2q: torch.Tensor,
) -> torch.Tensor:
    """Make unsigned and reduce the tensor."""
    return torch.ops.tiberate_he_ops.codec_rotate_make_unsigned_reduce_2q(
        a, perm, _2q
    )


def create_switcher_divide_by_p(
    c: torch.Tensor,
    p: torch.Tensor,
    _2q: torch.Tensor,
    Rs: torch.Tensor,
    PiRi: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> torch.Tensor:
    """Create a switcher by dividing by p."""
    return torch.ops.tiberate_he_ops.create_switcher_divide_by_p(
        c, p, _2q, Rs, PiRi, ql, qh, kl, kh
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
    """Rescale the tensor with exact rounding."""
    return torch.ops.tiberate_he_ops.rescale_exact_rounding_fused(
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
    """Rescale the tensor with non-exact rounding."""
    return torch.ops.tiberate_he_ops.rescale_non_exact_rounding_fused(
        a, Rs, rescaler, _2q, ql, qh, kl, kh
    )
