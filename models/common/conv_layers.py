
import torch
from torch import nn


# --------------------------------------------------------------------------------
class ConvMaxPoolBlock(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        ...

    # --------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Perform forward pass on the  input tensor

        Args:
        x: (torch.Tensor): Input Tensor

        """
        ...
