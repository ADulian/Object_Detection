import torch

# --------------------------------------------------------------------------------
def torch_epsilon(tensor: (torch.Tensor | None) = None) -> float:
    """Get a machine epsilon - the smallest representable number

    Args:
        tensor: (torch.Tensor | None): Torch tensor

    Returns:
        float: epsilon value
    """

    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Tensor must be of type torch.Tensor, current type: {type(tensor)}")

    dtype = torch.float32

    if tensor is not None:
        dtype = tensor.dtype

    return torch.finfo(dtype).eps

