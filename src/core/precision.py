SUPPORTED_TORCH_PRECISIONS = {"float16", "float32", "bfloat16"}


def torch_dtype_from_precision(precision: str):
    import torch

    if precision == "float16":
        return torch.float16
    if precision == "bfloat16":
        return torch.bfloat16
    if precision == "float32":
        return torch.float32
    raise ValueError(f"Unsupported precision={precision}. Supported values: {sorted(SUPPORTED_TORCH_PRECISIONS)}")
