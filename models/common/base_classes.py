
import torch
from torch import nn


# --------------------------------------------------------------------------------
class Conv2d13(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: tuple[int]):

        super().__init__()

        self.conv13 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=1),
                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels[1], kernel_size=3),)

    # --------------------------------------------------------------------------------
    def forward(self, x):
        return self.conv13(x)