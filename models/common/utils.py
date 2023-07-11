from typing import Callable

import torch.nn as nn

from ..common import basic_blocks

# --------------------------------------------------------------------------------
def get_layer(layer: str) -> Callable:
    """Get a layer from one of the modules

    Args:
        layer: (str): A layer name

    Returns:
        Callable: A reference to a layer

    """
    valid_modules = [nn, basic_blocks]

    for module in valid_modules:
        if hasattr(module, layer):
            return getattr(module, layer)

    raise ValueError(f"{layer} not found in any of the modules")

# --------------------------------------------------------------------------------
def auto_pad(kernel_size: int,
             padding: (int | None) = None) -> int:
    """Auto padding based on kernel size

    Args:
        kernel_size: (int): Kernel Size
        padding: (int | None): Padding, if None then the value will be computed from kernel

    Returns:
        int: Padding value

    """

    if padding is None:
        return kernel_size // 2
    else:
        return padding