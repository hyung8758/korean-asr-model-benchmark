def cuda_index(device: str) -> int:
    if ":" not in device:
        return 0
    return int(device.split(":", maxsplit=1)[1])


def validate_cuda_device(device: str) -> None:
    import torch

    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device was requested ({device}), but torch.cuda.is_available() is False. "
            "Check the NVIDIA driver and installed PyTorch CUDA build before running GPU decode."
        )
    device_count = torch.cuda.device_count()
    index = cuda_index(device)
    if index >= device_count:
        raise RuntimeError(f"CUDA device {device} was requested, but PyTorch sees only {device_count} CUDA devices.")


def validate_cuda_devices(gpu_ids: list[int]) -> None:
    import torch

    if not gpu_ids:
        raise ValueError("At least one GPU id is required in launcher mode.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU launcher mode was requested, but torch.cuda.is_available() is False. "
            "Check the NVIDIA driver and installed PyTorch CUDA build before running multi-GPU decode."
        )
    device_count = torch.cuda.device_count()
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= device_count]
    if missing:
        raise RuntimeError(f"Requested GPU ids {missing}, but PyTorch sees {device_count} CUDA devices.")
