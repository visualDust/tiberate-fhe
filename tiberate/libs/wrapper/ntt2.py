import torch


def ntt_radix2(
    a: torch.Tensor,
    even: torch.Tensor,
    odd: torch.Tensor,
    psi: torch.Tensor,
    _2q: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Perform NTT radix-2 on the input tensor."""
    return torch.ops.tiberate_ntt_ops.ntt_radix2(
        a, even, odd, psi, _2q, ql, qh, kl, kh
    )


def enter_ntt_radix2(
    a: torch.Tensor,
    Rs: torch.Tensor,
    even: torch.Tensor,
    odd: torch.Tensor,
    psi: torch.Tensor,
    _2q: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Prepare the tensor for NTT radix-2."""
    return torch.ops.tiberate_ntt_ops.enter_ntt_radix2(
        a, Rs, even, odd, psi, _2q, ql, qh, kl, kh
    )


def intt_radix2(
    a: torch.Tensor,
    even: torch.Tensor,
    odd: torch.Tensor,
    psi: torch.Tensor,
    Ninv: torch.Tensor,
    _2q: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Perform inverse NTT radix-2 on the input tensor."""
    return torch.ops.tiberate_ntt_ops.intt_radix2(
        a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh
    )


def intt_radix2_exit(
    a: torch.Tensor,
    even: torch.Tensor,
    odd: torch.Tensor,
    psi: torch.Tensor,
    Ninv: torch.Tensor,
    _2q: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Exit the inverse NTT radix-2."""
    return torch.ops.tiberate_ntt_ops.intt_radix2_exit(
        a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh
    )


def intt_radix2_exit_reduce(
    a: torch.Tensor,
    even: torch.Tensor,
    odd: torch.Tensor,
    psi: torch.Tensor,
    Ninv: torch.Tensor,
    _2q: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Exit and reduce the inverse NTT radix-2."""
    return torch.ops.tiberate_ntt_ops.intt_radix2_exit_reduce(
        a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh
    )


def intt_radix2_exit_reduce_signed(
    a: torch.Tensor,
    even: torch.Tensor,
    odd: torch.Tensor,
    psi: torch.Tensor,
    Ninv: torch.Tensor,
    _2q: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    kl: torch.Tensor,
    kh: torch.Tensor,
) -> None:
    """Exit, reduce, and sign the inverse NTT radix-2."""
    return torch.ops.tiberate_ntt_ops.intt_radix2_exit_reduce_signed(
        a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh
    )
