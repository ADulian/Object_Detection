from typing import Callable

import torch.nn as nn

# --------------------------------------------------------------------------------
def get_layer(layer: str) -> Callable:
    """Get a layer from one of the modules

    Args:
        layer: (str): A layer name

    Returns:
        Callable: A reference to a layer

    """
    valid_modules = [nn]

    for module in valid_modules:
        if hasattr(module, layer):
            return getattr(module, layer)

    raise ValueError(f"{layer} not found in any of the modules")
