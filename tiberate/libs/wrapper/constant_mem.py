import torch
from loguru import logger

from tiberate.libs.utils import constant_mem


def upload_constants_2qRsQlQhKlKh(
    _2q: list[torch.Tensor],
    Rs: list[torch.Tensor],
    ql: list[torch.Tensor],
    qh: list[torch.Tensor],
    kl: list[torch.Tensor],
    kh: list[torch.Tensor],
) -> None:
    """Upload constant memory tensors to the device."""
    logger.info(
        f"Uploading constant memory tensors for 2q, Rs, ql, qh, kl, kh for {len(_2q)} devices."
    )
    constant_mem.upload_constants_2qRsQlQhKlKh(_2q, Rs, ql, qh, kl, kh)


if __name__ == "__main__":

    def read_constants_2qRsQlQhKlKh(
        device: int, count: int
    ) -> tuple[torch.Tensor, ...]:
        """Read constants back from constant memory on the given device."""
        return tuple(
            constant_mem.test_read_constants_2qRsQlQhKlKh(device, count)
        )

    def make_rand_tensors(device_count: int, shape=(19,)) -> list[torch.Tensor]:
        return [
            torch.randint(
                low=1,
                high=1_000_000,
                size=shape,
                dtype=torch.int64,
                device=f"cuda:{i}",
            )
            for i in range(device_count)
        ]

    torch.manual_seed(42)  # For reproducibility
    device_count = torch.cuda.device_count()
    assert device_count > 0, "No CUDA devices available."

    # Generate separate tensors per field
    expected_per_field = {
        "_2q": make_rand_tensors(device_count),
        "Rs": make_rand_tensors(device_count),
        "ql": make_rand_tensors(device_count),
        "qh": make_rand_tensors(device_count),
        "kl": make_rand_tensors(device_count),
        "kh": make_rand_tensors(device_count),
    }

    # Upload to constant memory
    upload_constants_2qRsQlQhKlKh(
        _2q=expected_per_field["_2q"],
        Rs=expected_per_field["Rs"],
        ql=expected_per_field["ql"],
        qh=expected_per_field["qh"],
        kl=expected_per_field["kl"],
        kh=expected_per_field["kh"],
    )

    # Optional sync to ensure upload is complete before readback
    torch.cuda.synchronize()

    # Validate readback
    error_flag = False
    for device_id in range(device_count):
        readback = read_constants_2qRsQlQhKlKh(device_id, 19)
        for name, actual_tensor in zip(expected_per_field.keys(), readback):
            expected_tensor = expected_per_field[name][device_id]
            if not torch.allclose(actual_tensor.cpu(), expected_tensor.cpu()):
                logger.error(
                    f"\t ERROR! Mismatch on device {device_id}, field {name}"
                )
                logger.error(f"\t Expected: {expected_tensor.cpu()}")
                logger.error(f"\t Got     : {actual_tensor.cpu()}")
                error_flag = True
    if not error_flag:
        logger.success(
            "All constant memory tensors match expected values across devices."
        )
    else:
        logger.error(
            "Some constant memory tensors did not match expected values."
        )
