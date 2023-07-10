import torch
from torch import nn

from ..common.utils import get_layer

# --------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """A Convolution block with/without MaxPool
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 in_channels:int,
                 layers: list[dict]):
        """Convolution Block

        Args:
            in_channels: (int): Input channels
            layers: (list[dict]): A list of layers

        """
        super().__init__()
        self.block = self._init_block(in_channels=in_channels, layers=layers)

    # --------------------------------------------------------------------------------
    def _init_block(self,
                    in_channels: int,
                    layers: list[dict]) -> nn.Sequential:
        """Initialize block

        Args:
            in_channels: (int): Input channels
            layers: (list[dict]): A list of layers

        Returns:
            nn.Sequential: A sequential block

        """

        # Merge list into a single dict
        layers = {k:v
                  for d in layers
                  for k, v in d.items()}

        # Init
        block = []
        for layer_name, layer_kwargs in layers.items():
            # Get layer
            layer_name = layer_name.split("_")[0]
            layer = get_layer(layer=layer_name)

            # Update In channels
            if "conv" in layer_name.lower():
                layer_kwargs["in_channels"] = in_channels
                out_channels = layer_kwargs["out_channels"]
                in_channels = out_channels[-1] if isinstance(out_channels, list) else out_channels

            # Initialize and Append
            block.append(layer(**layer_kwargs))

        self.out_channels = in_channels

        return nn.Sequential(*block)

    # --------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass on the  input tensor

        Args:
            x: (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor

        """
        return self.block(x)
