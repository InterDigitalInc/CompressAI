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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import LayerNorm2d
from torch import Tensor

from compressai.layers import conv3x3
from compressai.models._helpers.mlic.utils import (
    checkerboard_anchor,
    checkerboard_merge,
    checkerboard_nonanchor,
)

__all__ = [
    "ContextReweighting",
    "GSCModule",
    "HGCPModule",
    "RoPE2D",
]


class _Gate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) * x


def _pointwise_then_dwconv(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(dim, dim, kernel_size=1),
        nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
    )


class ContextReweighting(nn.Module):
    """Channel-wise attention over an already captured spatial context."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.queries = nn.Conv2d(dim, dim, kernel_size=1)
        self.keys = nn.Conv2d(dim, dim, kernel_size=1)
        self.values = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_norm = LayerNorm2d(dim)
        self.gate = _Gate(dim)

    def channel_attention(self, input_tensor: Tensor) -> Tensor:
        batch_size, channels, height, width = input_tensor.shape
        num_positions = height * width
        normalized = self.norm(input_tensor)
        queries = self.queries(normalized).reshape(batch_size, channels, num_positions)
        keys = self.keys(normalized).reshape(batch_size, channels, num_positions)
        return F.softmax(
            queries @ keys.transpose(1, 2) * (num_positions**-0.5),
            dim=-1,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, channels, height, width = input_tensor.shape
        attention = self.channel_attention(input_tensor)
        normalized = self.norm(input_tensor)
        values = self.values(normalized).reshape(batch_size, channels, height * width)
        output = attention @ values
        output = output.reshape(batch_size, channels, height, width)
        output = self.proj(output)
        output = self.out_norm(output)
        return input_tensor + output + self.gate(output)


class RoPE2D(nn.Module):
    """Two-dimensional rotary position embedding for NCHW tensors."""

    def __init__(self, dim: int, learnable_thetas: bool = True) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE2D dim must be even")
        self.dim = int(dim)
        theta_x = torch.tensor(10000.0)
        theta_y = torch.tensor(10000.0)
        if learnable_thetas:
            self.theta_x = nn.Parameter(theta_x)
            self.theta_y = nn.Parameter(theta_y)
        else:
            self.register_buffer("theta_x", theta_x)
            self.register_buffer("theta_y", theta_y)
        freq = torch.arange(0, dim, 2, dtype=torch.float32) / float(dim)
        self.register_buffer("frequency", freq)

    def _angles(self, height: int, width: int, device: torch.device) -> Tensor:
        rows = torch.arange(height, device=device, dtype=self.frequency.dtype)
        cols = torch.arange(width, device=device, dtype=self.frequency.dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        theta_x = self.theta_x.to(device=device, dtype=self.frequency.dtype).abs()
        theta_y = self.theta_y.to(device=device, dtype=self.frequency.dtype).abs()
        inv_x = theta_x.clamp_min(1.0).pow(-self.frequency.to(device))
        inv_y = theta_y.clamp_min(1.0).pow(-self.frequency.to(device))
        return xx[..., None] * inv_x + yy[..., None] * inv_y

    def rotate(self, input_tensor: Tensor) -> Tensor:
        batch_size, channels, height, width = input_tensor.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {channels}")
        angles = self._angles(height, width, input_tensor.device).to(input_tensor.dtype)
        cos = angles.cos().permute(2, 0, 1).unsqueeze(0)
        sin = angles.sin().permute(2, 0, 1).unsqueeze(0)
        pairs = input_tensor.reshape(batch_size, channels // 2, 2, height, width)
        x_even = pairs[:, :, 0]
        x_odd = pairs[:, :, 1]
        rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            dim=2,
        )
        return rotated.reshape(batch_size, channels, height, width)

    def forward(
        self, query: Tensor, key: Optional[Tensor] = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        query = self.rotate(query)
        if key is None:
            return query
        return query, self.rotate(key)


class HGCPModule(nn.Module):
    """Hyperprior-guided global correlation prediction for the first slice."""

    def __init__(
        self,
        M: int,
        slice_ch: int,
        out_ch: Optional[int] = None,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        if slice_ch % num_heads != 0:
            raise ValueError("HGCPModule slice_ch must be divisible by num_heads")
        self.M = int(M)
        self.slice_ch = int(slice_ch)
        self.out_ch = int(out_ch or 2 * slice_ch)
        self.num_heads = int(num_heads)
        self.queries = _pointwise_then_dwconv(M)
        self.keys = _pointwise_then_dwconv(M)
        self.hyper_values = _pointwise_then_dwconv(M)
        self.slice_values = nn.Conv2d(slice_ch, M, kernel_size=1)
        self.proj = conv3x3(M, self.out_ch)
        self.gate = _Gate(self.out_ch)

    def _linear_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_size, channels, height, width = query.shape
        head_dim = channels // self.num_heads
        token_count = height * width
        outputs = []
        for index in range(self.num_heads):
            start = index * head_dim
            end = (index + 1) * head_dim
            query_i = query[:, start:end].reshape(batch_size, head_dim, token_count)
            key_i = key[:, start:end].reshape(batch_size, head_dim, token_count)
            value_i = value[:, start:end].reshape(batch_size, head_dim, token_count)
            key_i = F.softmax(key_i, dim=2)
            query_i = F.softmax(query_i, dim=1)
            context = key_i @ value_i.transpose(1, 2)
            outputs.append(
                (context.transpose(1, 2) @ query_i).reshape(
                    batch_size,
                    head_dim,
                    height,
                    width,
                )
            )
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        hyper_params: Tensor,
        anchor_y_hat: Optional[Tensor] = None,
    ) -> Tensor:
        hyper = hyper_params[:, : self.M]
        if anchor_y_hat is None:
            values = self.hyper_values(hyper)
        else:
            values = self.slice_values(anchor_y_hat)

        hyper_anchor = checkerboard_anchor(hyper)
        hyper_nonanchor = checkerboard_nonanchor(hyper)
        value_anchor = checkerboard_anchor(values)
        value_nonanchor = checkerboard_nonanchor(values)

        anchor_context = self._linear_attention(
            self.queries(hyper_anchor),
            self.keys(hyper_nonanchor),
            value_nonanchor,
        )
        nonanchor_context = self._linear_attention(
            self.queries(hyper_nonanchor),
            self.keys(hyper_anchor),
            value_anchor,
        )
        context = checkerboard_merge(anchor_context, nonanchor_context)
        context = self.proj(context)
        return context + self.gate(context)


class GSCModule(nn.Module):
    """Guided selective compression predictor compatible with leaf hooks."""

    def __init__(
        self,
        slice_ch: int,
        side_ch: Optional[int] = None,
        hidden_ch: Optional[int] = None,
        threshold: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_ch = int(hidden_ch or max(16, slice_ch))
        self.slice_ch = int(slice_ch)
        self.threshold = float(threshold)
        self.side_proj: nn.Module
        if side_ch is None:
            self.side_proj = nn.LazyConv2d(hidden_ch, kernel_size=1)
        else:
            self.side_proj = nn.Conv2d(side_ch, hidden_ch, kernel_size=1)
        self.predictor = nn.Sequential(
            conv3x3(hidden_ch + 3 * slice_ch, hidden_ch),
            nn.GELU(),
            conv3x3(hidden_ch, hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, slice_ch, kernel_size=1),
        )
        self.step_bias = nn.Parameter(torch.zeros(2, slice_ch, 1, 1))
        self.scale_slope = nn.Parameter(torch.tensor(8.0))

    def extra_repr(self) -> str:
        return f"slice_ch={self.slice_ch}, threshold={self.threshold}"

    def forward(
        self,
        *,
        side_params: Tensor,
        scales: Tensor,
        means: Tensor,
        step: str,
    ) -> Tensor | Dict[str, Tensor]:
        if step == "anchor":
            step_index = 0
        elif step == "non_anchor":
            step_index = 1
        else:
            raise ValueError(f'Invalid checkerboard step "{step}"')

        scale_prior = (scales >= self.threshold).to(scales.dtype)
        features = torch.cat(
            [
                self.side_proj(side_params),
                scales,
                means,
                scale_prior,
            ],
            dim=1,
        )
        logits = self.predictor(features)
        logits = logits + self.scale_slope * (scales - self.threshold)
        logits = logits + self.step_bias[step_index].unsqueeze(0)
        selective_map = torch.sigmoid(logits)
        return {
            "selective_map": selective_map,
            "scale_prior": scale_prior,
        }
