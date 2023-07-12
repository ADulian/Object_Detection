import yaml
from typing import Callable
from pathlib import Path

import torch.nn as nn


from models.common import basic_blocks
from utils.io import path_check

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
def get_cfg(root_path: (str | Path) ) -> dict:
    """Get config

    Args:
        root_path: (str | Path): Root path

    Returns:
        dict: model config
    """

    root_path = path_check(root_path)
    cfg = root_path / "config.yaml"

    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg