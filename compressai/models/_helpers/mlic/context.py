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

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.layers.attn.swin import Mlp

from .utils import (
    build_position_index,
    checkerboard_split,
    squeeze_anchor,
    squeeze_nonanchor,
    unsqueeze_anchor,
    unsqueeze_nonanchor,
)

__all__ = [
    "ChannelContext",
    "LinearGlobalInterContext",
    "LinearGlobalIntraContext",
    "LocalContext",
    "StackedCheckerboardConv",
    "VanillaGlobalInterContext",
    "VanillaGlobalIntraContext",
    "WindowCheckerboardAttn",
]


def _pointwise_then_dwconv(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
        nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
    )


def _checkerboard_coordinates(
    height: int,
    width: int,
    *,
    anchor: bool,
    device: torch.device,
) -> Tensor:
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    coords = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
    coords = coords.reshape(height * width, 2)
    parity = 1 if anchor else 0
    return coords[(coords[:, 0] + coords[:, 1]) % 2 == parity]


def _local_exclusion_mask(
    query_coords: Tensor,
    key_coords: Tensor,
    radius: int,
) -> Tensor:
    if radius < 0:
        return query_coords.new_zeros(
            (query_coords.shape[0], key_coords.shape[0]),
            dtype=torch.bool,
        )
    distance = (query_coords[:, None, :] - key_coords[None, :, :]).abs()
    return (distance <= radius).all(dim=-1)


def _quadratic_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    num_heads: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    batch_size, channels, query_count = queries.shape
    key_count = keys.shape[-1]
    head_dim = channels // num_heads

    query = queries.reshape(batch_size, num_heads, head_dim, query_count).transpose(
        -2,
        -1,
    )
    key = keys.reshape(batch_size, num_heads, head_dim, key_count).transpose(-2, -1)
    value = values.reshape(batch_size, num_heads, head_dim, key_count).transpose(
        -2,
        -1,
    )

    attention = (query @ key.transpose(-2, -1)) * (head_dim**-0.5)
    if attention_mask is not None:
        attention = attention.masked_fill(
            attention_mask.unsqueeze(0).unsqueeze(0),
            -100.0,
        )
    attention = F.softmax(attention, dim=-1)
    output = attention @ value
    return output.transpose(-2, -1).reshape(batch_size, channels, query_count)


class StackedCheckerboardConv(nn.Module):
    """Stacked convolutional local context used by MLIC."""

    def __init__(self, dim: int, kernel: int = 5, num_layers: int = 3) -> None:
        super().__init__()
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError(
                "StackedCheckerboardConv kernel must be a positive odd integer"
            )
        if num_layers <= 0 or num_layers % 2 == 0:
            raise ValueError(
                "StackedCheckerboardConv num_layers must be a positive odd"
            )

        layers: List[nn.Module] = []
        padding = kernel // 2
        for index in range(num_layers):
            out_channels = dim * 2 if index == num_layers - 1 else dim
            layers.append(
                nn.Conv2d(dim, out_channels, kernel_size=kernel, padding=padding)
            )
            if index != num_layers - 1:
                layers.append(nn.GELU())
        self.context = nn.Sequential(*layers)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.context(input_tensor)


class LocalContext(nn.Module):
    """Windowed intra-slice context used by MLIC++."""

    def __init__(
        self,
        dim: int = 32,
        window_size: int = 5,
        mlp_ratio: float = 2.0,
        num_heads: int = 2,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("LocalContext dim must be divisible by num_heads")

        self.H = -1
        self.W = -1
        self.dim = dim
        self.window_size = window_size
        self.window_area = window_size * window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(
            kernel_size=window_size,
            stride=1,
            padding=(window_size - 1) // 2,
        )
        self.relative_position_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim * 2, dim * 2)
        self.mlp = Mlp(
            in_features=dim * 2,
            hidden_features=int(dim * 2 * mlp_ratio),
            out_features=dim * 2,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.register_buffer(
            "relative_position_index",
            build_position_index((window_size, window_size)),
        )
        self.attn_mask: Optional[Tensor] = None
        self.fusion = nn.Conv2d(dim, dim * 2, kernel_size=window_size)

    def update_resolution(
        self,
        height: int,
        width: int,
        device: torch.device,
        mask: Optional[Tensor] = None,
    ) -> bool:
        updated = False
        if self.H != height or self.W != width:
            self.H = height
            self.W = width
            if mask is not None:
                self.attn_mask = mask.to(device)
                return True

            checkerboard = torch.zeros(
                (1, 2, height, width),
                device=device,
                requires_grad=False,
            )
            checkerboard[:, :, 0::2, 1::2] = 1
            checkerboard[:, :, 1::2, 0::2] = 1
            qk_windows = self.unfold(checkerboard).permute(0, 2, 1)
            qk_windows = qk_windows.reshape(
                1,
                height * width,
                2,
                1,
                self.window_size,
                self.window_size,
            ).permute(2, 0, 1, 3, 4, 5)
            q_windows, k_windows = qk_windows[0], qk_windows[1]
            query = q_windows.reshape(1, height * width, 1, self.window_area).permute(
                0,
                1,
                3,
                2,
            )
            key = k_windows.reshape(1, height * width, 1, self.window_area).permute(
                0,
                1,
                3,
                2,
            )
            attn_mask = query @ key.transpose(-2, -1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0.0, float(-100.0))
            self.attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)[0].detach()
            updated = True
        return updated

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, channels, height, width = input_tensor.shape
        num_tokens = height * width
        self.update_resolution(height, width, input_tensor.device)

        output = input_tensor.reshape(batch_size, channels, num_tokens).permute(0, 2, 1)
        output = self.norm1(output)

        qkv = self.qkv_proj(output).reshape(batch_size, height, width, 3, channels)
        qkv = qkv.permute(3, 0, 4, 1, 2).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2]

        qkv_windows = self.unfold(torch.cat([query, key, value], dim=1)).permute(
            0, 2, 1
        )
        qkv_windows = qkv_windows.reshape(
            batch_size,
            num_tokens,
            3,
            channels,
            self.window_size,
            self.window_size,
        ).permute(2, 0, 1, 3, 4, 5)
        query_windows, key_windows, value_windows = qkv_windows

        query = query_windows.reshape(
            batch_size,
            num_tokens,
            self.head_dim,
            self.num_heads,
            self.window_area,
        ).permute(0, 1, 3, 4, 2)
        key = key_windows.reshape(
            batch_size,
            num_tokens,
            self.head_dim,
            self.num_heads,
            self.window_area,
        ).permute(0, 1, 3, 4, 2)
        value = value_windows.reshape(
            batch_size,
            num_tokens,
            self.head_dim,
            self.num_heads,
            self.window_area,
        ).permute(0, 1, 3, 4, 2)

        attention = (query * self.scale) @ key.transpose(-2, -1)
        relative_position_bias = self.relative_position_table[
            self.relative_position_index.reshape(-1)
        ].view(self.window_area, self.window_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0).unsqueeze(1)

        if self.attn_mask is None:
            raise RuntimeError("LocalContext attention mask is not initialized")
        attention = attention + self.attn_mask.unsqueeze(0).unsqueeze(2)
        attention = self.softmax(attention)

        output = (attention @ value).reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.window_size,
            self.window_size,
            self.head_dim,
        )
        output = output.permute(0, 1, 3, 4, 2, 5).reshape(
            batch_size * num_tokens,
            self.window_size,
            self.window_size,
            channels,
        )
        output = output.permute(0, 3, 1, 2)
        output = self.fusion(output).reshape(batch_size, num_tokens, channels * 2)
        output = self.proj(output)
        output = output + self.mlp(self.norm2(output))
        return output.permute(0, 2, 1).reshape(batch_size, channels * 2, height, width)


class WindowCheckerboardAttn(LocalContext):
    """Overlapped window checkerboard attention used by MLIC+."""


class ChannelContext(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 192, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_dim * 4, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, channel_params: Tensor) -> Tensor:
        return self.fushion(channel_params)


class LinearGlobalIntraContext(nn.Module):
    def __init__(self, dim: int = 32, num_heads: int = 2) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "LinearGlobalIntraContext dim must be divisible by num_heads"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.keys = _pointwise_then_dwconv(dim)
        self.queries = _pointwise_then_dwconv(dim)
        self.values = _pointwise_then_dwconv(dim)
        self.reprojection = nn.Conv2d(dim, dim * 2, kernel_size=5, stride=1, padding=2)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(
                dim * 4,
                dim * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim * 4,
            ),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 2, kernel_size=1, stride=1),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        batch_size, _, height, width = x1.shape
        x1_anchor, x1_nonanchor = checkerboard_split(x1)
        queries = squeeze_nonanchor(self.queries(x1_nonanchor)).reshape(
            batch_size,
            self.dim,
            height * width // 2,
        )
        keys = squeeze_anchor(self.keys(x1_anchor)).reshape(
            batch_size,
            self.dim,
            height * width // 2,
        )
        values = squeeze_anchor(self.values(x2)).reshape(
            batch_size,
            self.dim,
            height * width // 2,
        )
        head_dim = self.dim // self.num_heads

        attended_values = []
        for index in range(self.num_heads):
            key = F.softmax(
                keys[:, index * head_dim : (index + 1) * head_dim, :], dim=2
            )
            query = F.softmax(
                queries[:, index * head_dim : (index + 1) * head_dim, :],
                dim=1,
            )
            value = values[:, index * head_dim : (index + 1) * head_dim, :]
            key = unsqueeze_anchor(
                key.reshape(batch_size, head_dim, height, width // 2)
            )
            key = key.reshape(batch_size, head_dim, height * width)
            value = unsqueeze_anchor(
                value.reshape(batch_size, head_dim, height, width // 2)
            )
            value = value.reshape(batch_size, head_dim, height * width)
            query = unsqueeze_nonanchor(
                query.reshape(batch_size, head_dim, height, width // 2)
            )
            query = query.reshape(batch_size, head_dim, height * width)
            context = key @ value.transpose(1, 2)
            attended_values.append(
                (context.transpose(1, 2) @ query).reshape(
                    batch_size, head_dim, height, width
                )
            )

        attention = self.reprojection(torch.cat(attended_values, dim=1))
        return attention + self.mlp(attention)


class LinearGlobalInterContext(nn.Module):
    def __init__(self, dim: int = 32, out_dim: int = 64, num_heads: int = 2) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "LinearGlobalInterContext dim must be divisible by num_heads"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.keys = _pointwise_then_dwconv(dim)
        self.queries = _pointwise_then_dwconv(dim)
        self.values = _pointwise_then_dwconv(dim)
        self.reprojection = nn.Conv2d(
            dim,
            out_dim * 3 // 2,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim * 3 // 2, out_dim * 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(
                out_dim * 2,
                out_dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_dim * 2,
            ),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, stride=1),
        )
        self.skip = nn.Conv2d(out_dim * 3 // 2, out_dim, kernel_size=1, stride=1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.shape
        queries = self.queries(input_tensor).reshape(
            batch_size, self.dim, height * width
        )
        keys = self.keys(input_tensor).reshape(batch_size, self.dim, height * width)
        values = self.values(input_tensor).reshape(batch_size, self.dim, height * width)
        head_dim = self.dim // self.num_heads

        attended_values = []
        for index in range(self.num_heads):
            key = F.softmax(
                keys[:, index * head_dim : (index + 1) * head_dim, :], dim=2
            )
            query = F.softmax(
                queries[:, index * head_dim : (index + 1) * head_dim, :],
                dim=1,
            )
            value = values[:, index * head_dim : (index + 1) * head_dim, :]
            context = key @ value.transpose(1, 2)
            attended_values.append(
                (context.transpose(1, 2) @ query).reshape(
                    batch_size, head_dim, height, width
                )
            )

        attention = self.reprojection(torch.cat(attended_values, dim=1))
        return self.skip(attention) + self.mlp(attention)


class VanillaGlobalIntraContext(nn.Module):
    """Quadratic intra-slice global context used by MLIC and MLIC+."""

    def __init__(
        self,
        dim: int = 32,
        num_heads: int = 2,
        local_mask_radius: int = 2,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "VanillaGlobalIntraContext dim must be divisible by num_heads"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.local_mask_radius = local_mask_radius
        self.keys = _pointwise_then_dwconv(dim)
        self.queries = _pointwise_then_dwconv(dim)
        self.values = _pointwise_then_dwconv(dim)
        self.reprojection = nn.Conv2d(dim, dim * 2, kernel_size=5, stride=1, padding=2)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(
                dim * 4,
                dim * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim * 4,
            ),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 2, kernel_size=1, stride=1),
        )

    def _attention_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        query_coords = _checkerboard_coordinates(
            height,
            width,
            anchor=False,
            device=device,
        )
        key_coords = _checkerboard_coordinates(
            height,
            width,
            anchor=True,
            device=device,
        )
        return _local_exclusion_mask(query_coords, key_coords, self.local_mask_radius)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        batch_size, _, height, width = x1.shape
        x1_anchor, x1_nonanchor = checkerboard_split(x1)
        queries = squeeze_nonanchor(self.queries(x1_nonanchor)).reshape(
            batch_size,
            self.dim,
            height * width // 2,
        )
        keys = squeeze_anchor(self.keys(x1_anchor)).reshape(
            batch_size,
            self.dim,
            height * width // 2,
        )
        values = squeeze_anchor(self.values(x2)).reshape(
            batch_size,
            self.dim,
            height * width // 2,
        )

        attention_mask = self._attention_mask(height, width, x1.device)
        attention = _quadratic_attention(
            queries,
            keys,
            values,
            self.num_heads,
            attention_mask,
        )
        attention = unsqueeze_nonanchor(
            attention.reshape(batch_size, self.dim, height, width // 2)
        )
        attention = self.reprojection(attention)
        return attention + self.mlp(attention)


class VanillaGlobalInterContext(nn.Module):
    """Quadratic inter-slice global context used by MLIC+."""

    def __init__(
        self,
        in_dim: int = 32,
        out_dim: int = 64,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        if in_dim % num_heads != 0:
            raise ValueError(
                "VanillaGlobalInterContext in_dim must be divisible by num_heads"
            )

        self.dim = in_dim
        self.num_heads = num_heads
        self.keys = _pointwise_then_dwconv(in_dim)
        self.queries = _pointwise_then_dwconv(in_dim)
        self.values = _pointwise_then_dwconv(in_dim)
        self.reprojection = nn.Conv2d(
            in_dim,
            out_dim * 3 // 2,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim * 3 // 2, out_dim * 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(
                out_dim * 2,
                out_dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_dim * 2,
            ),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, stride=1),
        )
        self.skip = nn.Conv2d(out_dim * 3 // 2, out_dim, kernel_size=1, stride=1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.shape
        num_tokens = height * width
        queries = self.queries(input_tensor).reshape(batch_size, self.dim, num_tokens)
        keys = self.keys(input_tensor).reshape(batch_size, self.dim, num_tokens)
        values = self.values(input_tensor).reshape(batch_size, self.dim, num_tokens)

        attention = _quadratic_attention(queries, keys, values, self.num_heads)
        attention = attention.reshape(batch_size, self.dim, height, width)
        attention = self.reprojection(attention)
        return self.skip(attention) + self.mlp(attention)
