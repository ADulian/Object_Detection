import torch

from torch import nn

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

# --------------------------------------------------------------------------------
class Conv2dBasic(nn.Module):
    """A Basic Convolution Block with Batch Norm and ReLU
    """
    # --------------------------------------------------------------------------------
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: (int | None) = None,
                 bias: bool = False):
        """Initialize Block

        Args:
            in_channels: (int): Input channels
            out_channels: (int): Output channels
            kernel_size: (int): Kernel size
            stride: (int): Stride
            padding: (int | None): Padding, if None then it will be automatically computed from kernel size
            bias: (bool): Bias, if false then a Batch Norm will be added

        """

        super().__init__()

        # Conv Layer
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=auto_pad(kernel_size=kernel_size, padding=padding),
                            bias=bias)]

        # Batch Norm Layer
        if not bias:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # Activation
        layers.append(nn.ReLU())

        # Init Sequential block
        self.block = nn.Sequential(*layers)

    # --------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass on the  input tensor

        Args:
            x: (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor

        """
        return self.block(x)

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

        self.conv13 = nn.Sequential(Conv2dBasic(in_channels=in_channels, out_channels=out_channels[0],
                                                kernel_size=1),
                                    Conv2dBasic(in_channels=out_channels[0], out_channels=out_channels[1],
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
