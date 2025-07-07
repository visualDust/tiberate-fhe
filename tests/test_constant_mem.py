import torch

from tiberate.context.constant_mem_context import (
    TensorConstLayout,
    TensorEntry,
    read_constant_chunk,
    upload_constant_2qRsQlQhKlKhNinv,
)


def make_tensor_entry(
    name: str, size: int, dtype: torch.dtype, device: torch.device
) -> TensorEntry:
    return TensorEntry(
        tensor=torch.randint(1, 1_000_000, (size,), dtype=dtype, device=device),
        name=name,
    )


def upload_and_read_single_device(
    device: torch.device, layout_mode: str  # "left" | "right"
):
    torch.manual_seed(42)

    entries = [
        make_tensor_entry("_2q", 32, torch.int64, device),
        make_tensor_entry("Rs", 32, torch.int32, device),
        make_tensor_entry("ql", 32, torch.int64, device),
        make_tensor_entry("qh", 32, torch.int32, device),
        make_tensor_entry("kl", 32, torch.int64, device),
        make_tensor_entry("kh", 32, torch.int32, device),
        make_tensor_entry("Ninv", 32, torch.int64, device),
    ]

    plan = TensorConstLayout(entries=entries, layout_mode=layout_mode)
    offsets = plan.upload()

    print(f"Device {device} uploaded constants with offsets:")
    for entry, offset in zip(plan.entries, offsets):
        size = entry.tensor.numel() * entry.tensor.element_size()
        if layout_mode == "right":
            physical_offset = plan.max_bytes - offset - size
        else:
            physical_offset = offset

        print(
            f"  {entry.name:<5} | logical: [{offset:5}, {offset + size:5})"
            f"  → physical: [{physical_offset:5}, {physical_offset + size:5})"
        )

    torch.cuda.synchronize()

    error_flag = False
    for entry, offset in zip(plan.entries, offsets):
        actual = read_constant_chunk(
            device=0,
            offset_bytes=offset,
            count=entry.tensor.numel(),
            dtype=entry.tensor.dtype,
            layout=plan.layout_mode,
        )
        expected = entry.tensor.cpu()
        if not torch.allclose(actual.cpu(), expected):
            print(
                f"Mismatch in {entry.name} ({entry.tensor.dtype}):\n"
                f"Expected: {expected}\nGot     : {actual}"
            )
            error_flag = True
    if error_flag:
        raise AssertionError(
            f"Device {device} Some constants did not match after upload/readback."
        )
    else:
        print(f"Device {device} All constants verified successfully.")


def test_upload_and_read():
    devices = [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ]
    layout_modes = ["left", "right"]

    for device in devices:
        for layout_mode in layout_modes:
            print(f"Testing device {device} with layout '{layout_mode}'")
            upload_and_read_single_device(device, layout_mode)
            print(f"Completed device {device} with layout '{layout_mode}'\n")


def make_rand_tensors(device_count: int, shape=(32,)):
    return [
        torch.randint(
            low=1_000,
            high=1_000_000,
            size=shape,
            dtype=torch.int64,
            device=f"cuda:{i}",
        )
        for i in range(device_count)
    ]


def test_upload_constants_2qRsQlQhKlKhNinv():
    print("Testing upload_constants_2qRsQlQhKlKhNinv...")
    _2q_list = make_rand_tensors(torch.cuda.device_count())
    Rs_list = make_rand_tensors(torch.cuda.device_count())
    ql_list = make_rand_tensors(torch.cuda.device_count())
    qh_list = make_rand_tensors(torch.cuda.device_count())
    kl_list = make_rand_tensors(torch.cuda.device_count())
    kh_list = make_rand_tensors(torch.cuda.device_count())
    Ninv_list = make_rand_tensors(torch.cuda.device_count())

    upload_constant_2qRsQlQhKlKhNinv(
        _2q=_2q_list,
        Rs=Rs_list,
        ql=ql_list,
        qh=qh_list,
        kl=kl_list,
        kh=kh_list,
        Ninv=Ninv_list,
        gravity="right",
        verbose=True,
    )
