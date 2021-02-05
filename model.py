import torch
from torch import nn, Tensor
# noinspection PyPep8Naming
from torch.nn import functional as F


class ImageTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4, padding_mode='reflect'),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )

        self._down_blocks = nn.Sequential(
            DownBlock(32, 64, kernel_size=3),
            DownBlock(64, 128, kernel_size=3),
        )

        self._residual_blocks = nn.Sequential(
            *[ResidualBlock(128, kernel_size=3) for _ in range(5)]
        )

        self._up_blocks = nn.Sequential(
            UpBlock(128, 64, kernel_size=3),
            UpBlock(64, 32, kernel_size=3),
        )

        self._final = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4, padding_mode='reflect')

    def forward(self, x: Tensor) -> Tensor:
        x = self._initial(x)
        x = self._down_blocks(x)
        x = self._residual_blocks(x)
        x = self._up_blocks(x)
        x = self._final(x)
        x = torch.sigmoid(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            padding_mode='reflect'
        )

        self._norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self._conv(x)
        x = self._norm(x)
        x = F.relu(x, inplace=True)
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self._conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode='reflect'
        )

        self._norm1 = nn.InstanceNorm2d(channels, affine=True)

        self._conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode='reflect'
        )

        self._norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self._conv1(x)
        x = self._norm1(x)
        x = F.relu(x, inplace=True)
        x = self._conv2(x)
        x = self._norm2(x)
        x = x + residual
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode='reflect'
        )

        self._norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self._conv(x)
        x = self._norm(x)
        x = F.relu(x, inplace=True)
        return x
