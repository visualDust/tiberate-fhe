from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from tiberate.libs.wrapper import const_pool

LayoutGravity = Literal["left", "right"]


def upload_tensor_list(
    tensors: list[torch.Tensor],
    offsets: list[int],
    layout: LayoutGravity = "left",
    device_id: int = 0,
) -> None:
    """Upload a list of tensors to constant memory with specified offsets."""
    layout_flag = 0 if layout == "left" else 1
    const_pool.upload_tensor_list(tensors, offsets, layout_flag, device_id)


def read_constant_chunk(
    device: int,
    offset_bytes: int,
    count: int,
    dtype: torch.dtype,
    layout: LayoutGravity = "left",
) -> torch.Tensor:
    """Read a chunk of constants from constant memory."""
    dummy = torch.empty(0, device=f"cuda:{device}")
    layout_flag = 0 if layout == "left" else 1
    return const_pool.read_constant_chunk(
        dummy, offset_bytes, count, dtype, layout_flag
    )


@dataclass
class TensorEntry:
    tensor: torch.Tensor
    name: str = ""  # Optional name for debugging


@dataclass
class TensorConstLayout:
    entries: list[TensorEntry]
    gravity: LayoutGravity = "left"
    align_bytes: int = 8  # Alignment for safety (especially with int64)
    max_bytes: int = 64 * 1024

    def compute_offsets(self) -> list[int]:
        """Compute byte offsets for each tensor, respecting alignment."""
        offsets = []
        offset = 0
        for entry in self.entries:
            size = entry.tensor.numel() * entry.tensor.element_size()
            # Align offset
            offset = (
                (offset + self.align_bytes - 1)
                // self.align_bytes
                * self.align_bytes
            )
            if offset + size > self.max_bytes:
                raise RuntimeError(
                    f"Constant memory overflow: {offset + size} > {self.max_bytes}"
                )
            offsets.append(offset)
            offset += size
        return offsets

    def upload(self):
        self.offsets = self.compute_offsets()
        tensors = [entry.tensor for entry in self.entries]
        device_id = tensors[0].device.index
        upload_tensor_list(tensors, self.offsets, self.gravity, device_id)
        torch.cuda.synchronize()  # Ensure upload is complete
        return self.offsets  # For debugging or readback

    def validate(self):
        """
        Check if the content of constant memory matches the expected.
        """
        for entry, offset in zip(self.entries, self.offsets):
            actual = read_constant_chunk(
                device=entry.tensor.device.index,
                offset_bytes=offset,
                count=entry.tensor.numel(),
                dtype=entry.tensor.dtype,
                layout=self.gravity,
            )
            if not torch.allclose(actual.cpu(), entry.tensor.cpu()):
                # raise AssertionError(
                #     f"Constant memory mismatch for {entry.name}: "
                #     f"expected {entry.tensor}, got {actual}"
                # )
                return False  # Return False if any mismatch occurs
        return True  # Return True if all entries match


def make_padded(t: torch.Tensor, length: int, pad_left: bool) -> torch.Tensor:
    """
    Pad the last dimension of tensor t to the given length.

    Args:
        t (torch.Tensor): The tensor to pad.
        length (int): The target length for the last dimension.
        pad_left (bool): If True, pad on the left; otherwise, pad on the right.

    Returns:
        torch.Tensor: The padded tensor.
    """
    current_len = t.shape[-1]
    if current_len > length:
        raise ValueError(
            f"Tensor is longer ({current_len}) than static limit ({length})"
        )
    if current_len == length:
        return t.clone()
    pad_amount = length - current_len
    # Pad format is (last_dim_pad_right, last_dim_pad_left)
    pad = (pad_amount, 0) if pad_left else (0, pad_amount)
    padded = F.pad(t, pad, value=0)
    return padded


def upload_constant_2qRsQlQhKlKhNinv(
    _2q: list[torch.Tensor],  # [2q_dev0, 2q_dev1, ...]
    Rs: list[torch.Tensor],  # [Rs_dev0, Rs_dev1, ...]
    ql: list[torch.Tensor],
    qh: list[torch.Tensor],
    kl: list[torch.Tensor],
    kh: list[torch.Tensor],
    Ninv: list[torch.Tensor],
    gravity: LayoutGravity = "right",
    verbose: bool = False,
) -> None:
    """
        Upload constants to constant memory across multiple devices.

        Args: _2q, Rs, ql, qh, kl, kh, Ninv: Lists of tensors for each device.
                layout: "left" or "right" for constant memory layout.

        Raises:
            ValueError: If tensors are not on the same device.
            RuntimeError: If constant memory overflow occurs.
            AssertionError: If readback does not match uploaded tensors.

        My design here are:
            - use right gravity for those constants since in CUDA kernel they are dropped from left to right, exampe:
                - level 0: [0,1,2,3,4,5,6,...]
                - level 1: [1,2,3,4,5,6,...]
                - level 2: [2,3,4,5,6,...]
                - so the offset for _2q in CUDA kernel [2q_end_offset-gridDim.x+ blockIdx.x]
            - pad each constant tensor to length 128 (which is engough for most cases), so the constant offset will be aligned to 128*sizeof(int32/int64) for sure, and Python won't have to tell CUDA what is the length of rns since CUDA will use the same static offset too. todo)) chat with PuQing and see which design is better, tell the rns length or pad to static length. like:
                - level 0: [0,1,2,3,4,5,6, 0,0,0,0,0,0,0,0,...]  # padded to 128
                - level 1: [1,2,3,4,5,6, 0,0,0,0,0,0,0,0,...,0]  # padded to 128
                - level 2: [2,3,4,5,6, 0,0,0,0,0,0,0,0,...,0,0] # padded to 128
            - use 64KB constant memory for each device, which is enough for most cases.
            - use 8 bytes alignment for each constant tensor, which is safe for all tensor types (int32, int64, float32, float64, etc.).
            - use a dataclass to represent each tensor entry, in case in the future we need to manage multiple context of constant memory. And we can implement a with statement to manage the context automatically.

    # Example Layouts:
    #  Gravity: Left
    # ┌────────────────────────────────────────────────────────────────────────────────────────┐
    # │┌────────────────┐┌────────────────┐┌────────────────┐                                  │
    # ││_2q,..,0,0,0,0,0││Rs ,..,0,0,0,0,0││ql ,..,0,0,0,0,0│...                               │
    # │└────────────────┘└────────────────┘└────────────────┘                                  │
    # └────────────────────────────────────────────────────────────────────────────────────────┘
    #  Gravity: Right
    # ┌────────────────────────────────────────────────────────────────────────────────────────┐
    # │                              ┌────────────────┐┌────────────────┐┌────────────────┐    │
    # │                              │0,0,0,0,0,.., ql││0,0,0,0,0 ,..,Rs││0,0,0,0,0,..,_2q│... │
    # │                              └────────────────┘└────────────────┘└────────────────┘    │
    # └────────────────────────────────────────────────────────────────────────────────────────┘
    """
    REGION_LEN = 128

    device_count = len(_2q)
    layout_on_devices: list[tuple[int, TensorConstLayout, list[int]]] = []
    plans = []

    for i in range(device_count):
        dev_id = _2q[i].device.index
        # should be on the same device
        if not all(
            t.device.index == dev_id
            for t in [_2q[i], Rs[i], ql[i], qh[i], kl[i], kh[i], Ninv[i]]
        ):
            raise ValueError(f"All tensors must be on the same device {i}")
        # should have the same shape
        if not all(
            t.shape == _2q[i].shape
            for t in [Rs[i], ql[i], qh[i], kl[i], kh[i], Ninv[i]]
        ):
            raise ValueError("All tensors must have the same shape")
        # should be on the same dtype
        if not all(
            t.dtype == _2q[i].dtype
            for t in [Rs[i], ql[i], qh[i], kl[i], kh[i], Ninv[i]]
        ):
            raise ValueError("All tensors must have the same dtype")
        pad_left = gravity == "right"  # Right layout -> pad 0 on the left side
        padded_entries = [
            TensorEntry(
                make_padded(_2q[i], REGION_LEN, pad_left),
                "_2q",
            ),
            TensorEntry(
                make_padded(Rs[i], REGION_LEN, pad_left),
                "Rs",
            ),
            TensorEntry(
                make_padded(ql[i], REGION_LEN, pad_left),
                "ql",
            ),
            TensorEntry(
                make_padded(qh[i], REGION_LEN, pad_left),
                "qh",
            ),
            TensorEntry(
                make_padded(kl[i], REGION_LEN, pad_left),
                "kl",
            ),
            TensorEntry(
                make_padded(kh[i], REGION_LEN, pad_left),
                "kh",
            ),
            TensorEntry(
                make_padded(Ninv[i], REGION_LEN, pad_left),
                "Ninv",
            ),
        ]

        plan = TensorConstLayout(padded_entries, gravity=gravity)
        offsets = plan.upload()
        plans.append(plan)
        if verbose:
            layout_on_devices.append((dev_id, plan, offsets))

        # the function ends here. The following are pure debugging prints.
        if verbose:
            print(f"Device {dev_id} uploaded constants with offsets:")
            for entry, offset in zip(plan.entries, offsets):
                size = entry.tensor.numel() * entry.tensor.element_size()
                if gravity == "right":
                    physical_offset = plan.max_bytes - offset - size
                else:
                    physical_offset = offset

                print(
                    f"  {entry.name:<5} ({entry.tensor.shape}) | logical: [{offset:5}, {offset + size:5})"
                    f"  → physical: [{physical_offset:5}, {physical_offset + size:5})"
                )

    if verbose:
        # debug readback
        error_flag = False
        for dev_id, plan, offsets in layout_on_devices:
            for entry, offset in zip(plan.entries, offsets):
                actual = read_constant_chunk(
                    device=dev_id,
                    offset_bytes=offset,
                    count=entry.tensor.numel(),
                    dtype=entry.tensor.dtype,
                    layout=plan.gravity,
                )
                expected = entry.tensor.cpu()
                if not torch.allclose(actual.cpu(), expected):
                    print(
                        f"Constant memory context mismatch in {entry.name} ({entry.tensor.dtype}):\n"
                        f"Expected: {expected}\nGot     : {actual}"
                    )
                    error_flag = True
        if error_flag:
            raise AssertionError(
                "Some constants did not match after upload/readback."
            )
        else:
            print("All constants verified successfully across devices.")

    return plans  # Return the plans for further use or validation
