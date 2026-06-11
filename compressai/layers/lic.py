"""Shared learned-image-compression building blocks."""

from __future__ import annotations

import torch.nn as nn

from torch import Tensor

from compressai.models.sensetime import ResidualBottleneckBlock
from compressai.models.utils import conv, deconv

__all__ = [
    "ResidualBottleneckBlockWithStride",
    "ResidualBottleneckBlockWithUpsample",
]


class ResidualBottleneckBlockWithStride(nn.Module):
    """Stride-2 5x5 conv followed by three residual bottleneck blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = conv(in_ch, out_ch, kernel_size=5, stride=2)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.conv(input_tensor)
        output = self.res1(output)
        output = self.res2(output)
        return self.res3(output)


class ResidualBottleneckBlockWithUpsample(nn.Module):
    """Three residual bottleneck blocks followed by a stride-2 5x5 deconv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = deconv(in_ch, out_ch, kernel_size=5, stride=2)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.res1(input_tensor)
        output = self.res2(output)
        output = self.res3(output)
        return self.conv(output)
