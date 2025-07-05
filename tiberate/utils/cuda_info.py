from tiberate.libs.utils import cuda_info


def format_bytes(n):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def get_cuda_info():
    info_dict = cuda_info.get_cuda_device_properties()
    return info_dict


if __name__ == "__main__":
    cuda_info = get_cuda_info()
    for device, properties in cuda_info.items():
        print(f"Device {device}:")
        for key, value in properties.items():
            print(f"  {key}: {value}")
