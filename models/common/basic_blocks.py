import torch

from torch import nn


# --------------------------------------------------------------------------------
class Conv2d13(nn.Module):
    """A Convolution Block with the following structure:

    [0] - Conv 1x1
    [1] - Conv 3x3

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: tuple[int]):
        """Initialize Block

        Args:
            in_channels: (int): Input channels
            out_channels: (tuple[int]): Output Channels

        """

        super().__init__()

        self.conv13 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0],
                                              kernel_size=1),
                                    nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1],
                                              kernel_size=3),)

    # --------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass on the  input tensor

        Args:
            x: (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor

        """
        return self.conv13(x)