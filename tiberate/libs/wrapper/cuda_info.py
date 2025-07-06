from rich.console import Console

from tiberate.libs.utils import cuda_info

console = Console()


def get_cuda_info():
    return cuda_info.get_cuda_device_properties()


if __name__ == "__main__":
    info = get_cuda_info()
    print(info)
