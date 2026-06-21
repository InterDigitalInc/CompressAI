# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/JiangWeibeta/MLIC
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

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers import (
    GDN,
    conv1x1,
    conv3x3,
    subpel_conv3x3,
)

__all__ = [
    "AnalysisTransform",
    "EntropyParameters",
    "HyperAnalysis",
    "HyperSynthesis",
    "LatentResidualPrediction",
    "SynthesisTransform",
]


class _ResidualBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.act = nn.GELU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        self.skip = (
            conv1x1(in_ch, out_ch, stride=stride)
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = input_tensor
        output = self.gdn(self.conv2(self.act(self.conv1(input_tensor))))
        if self.skip is not None:
            identity = self.skip(input_tensor)
        return output + identity


class _ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2) -> None:
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.act = nn.GELU()
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.subpel_conv(input_tensor)
        output = self.igdn(self.conv(self.act(output)))
        return output + self.upsample(input_tensor)


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.act = nn.GELU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else None

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = input_tensor
        output = self.act(self.conv2(self.act(self.conv1(input_tensor))))
        if self.skip is not None:
            identity = self.skip(input_tensor)
        return output + identity


class AnalysisTransform(nn.Module):
    """MLIC++ analysis transform."""

    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.analysis_transform = nn.Sequential(
            _ResidualBlockWithStride(3, N, stride=2),
            _ResidualBlock(N, N),
            _ResidualBlockWithStride(N, N, stride=2),
            _ResidualBlock(N, N),
            _ResidualBlockWithStride(N, N, stride=2),
            _ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.analysis_transform(input_tensor)


class HyperAnalysis(nn.Module):
    def __init__(self, M: int = 192, N: int = 192) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.reduction(input_tensor)


class HyperSynthesis(nn.Module):
    def __init__(self, M: int = 192, N: int = 192) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.increase = nn.Sequential(
            conv3x3(N, M),
            nn.GELU(),
            subpel_conv3x3(M, M, 2),
            nn.GELU(),
            conv3x3(M, M * 3 // 2),
            nn.GELU(),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.GELU(),
            conv3x3(M * 3 // 2, M * 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.increase(input_tensor)


class SynthesisTransform(nn.Module):
    """MLIC++ synthesis transform."""

    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            _ResidualBlock(M, M),
            _ResidualBlockUpsample(M, N, 2),
            _ResidualBlock(N, N),
            _ResidualBlockUpsample(N, N, 2),
            _ResidualBlock(N, N),
            _ResidualBlockUpsample(N, N, 2),
            _ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.synthesis_transform(input_tensor)


class EntropyParameters(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, 320, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, params: Tensor) -> Tensor:
        return self.fusion(params)


class LatentResidualPrediction(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lrp_transform = nn.Sequential(
            conv3x3(in_dim, 224),
            act(),
            conv3x3(224, 128),
            act(),
            conv3x3(128, out_dim),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return 0.5 * torch.tanh(self.lrp_transform(input_tensor))
