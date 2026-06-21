# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts the MLIC family design from https://github.com/JiangWeibeta/MLIC
# (originally distributed under the Apache License 2.0). Modifications by
# InterDigital Communications, Inc. are released under the BSD 3-Clause Clear
# License terms below.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import Type

import torch.nn as nn

from timm.layers import LayerNorm2d
from torch import Tensor

from compressai.layers import conv3x3, subpel_conv3x3
from compressai.models._helpers.mlic.transforms import (
    _ResidualBlockUpsample,
    _ResidualBlockWithStride,
)

__all__ = [
    "STMAnalysis",
    "STMSynthesis",
    "SimpleTokenMixing",
]


class _DepthwiseResidualBlock(nn.Module):
    def __init__(self, dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            act(),
            nn.Conv2d(dim, dim, kernel_size=1),
            act(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class _Gate(nn.Module):
    def __init__(self, dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            act(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            act(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.gate(x) * x


class SimpleTokenMixing(nn.Module):
    """MetaFormer-style token mixing block used by MLICv2 transforms."""

    def __init__(self, dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.token_mixer = nn.Sequential(
            _DepthwiseResidualBlock(dim, act=act),
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.norm2 = LayerNorm2d(dim)
        self.gate = _Gate(dim, act=act)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.token_mixer(self.norm1(x))
        return x + self.gate(self.norm2(x))


def _stm_pair(dim: int) -> nn.Sequential:
    return nn.Sequential(SimpleTokenMixing(dim), SimpleTokenMixing(dim))


class STMAnalysis(nn.Module):
    """MLICv2 analysis transform with STM blocks replacing residual blocks."""

    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.analysis_transform = nn.Sequential(
            _ResidualBlockWithStride(3, N, stride=2),
            _stm_pair(N),
            _ResidualBlockWithStride(N, N, stride=2),
            _stm_pair(N),
            _ResidualBlockWithStride(N, N, stride=2),
            _stm_pair(N),
            conv3x3(N, M, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.analysis_transform(input_tensor)


class STMSynthesis(nn.Module):
    """MLICv2 synthesis transform with STM blocks replacing residual blocks."""

    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            _stm_pair(M),
            _ResidualBlockUpsample(M, N, 2),
            _stm_pair(N),
            _ResidualBlockUpsample(N, N, 2),
            _stm_pair(N),
            _ResidualBlockUpsample(N, N, 2),
            _stm_pair(N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.synthesis_transform(input_tensor)
