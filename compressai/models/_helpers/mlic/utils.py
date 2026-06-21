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

from typing import Callable, List, Sequence, Tuple, Union

import torch

from torch import Tensor

from compressai.entropy_models import EntropyModel

__all__ = [
    "build_position_index",
    "checkerboard_anchor",
    "checkerboard_merge",
    "checkerboard_nonanchor",
    "checkerboard_split",
    "compress_symbols",
    "decompress_symbols",
    "squeeze_anchor",
    "squeeze_nonanchor",
    "unsqueeze_anchor",
    "unsqueeze_nonanchor",
]


def build_position_index(window_size: Union[int, Tuple[int, int]]) -> Tensor:
    if isinstance(window_size, int):
        window_height = window_width = window_size
    else:
        window_height, window_width = window_size

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(window_height),
            torch.arange(window_width),
            indexing="ij",
        )
    )
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_height - 1
    relative_coords[:, :, 1] += window_width - 1
    relative_coords[:, :, 0] *= 2 * window_width - 1
    return relative_coords.sum(-1)


def checkerboard_split(input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
    return checkerboard_anchor(input_tensor), checkerboard_nonanchor(input_tensor)


def checkerboard_merge(anchor: Tensor, nonanchor: Tensor) -> Tensor:
    return anchor + nonanchor


def checkerboard_anchor(input_tensor: Tensor) -> Tensor:
    output = torch.zeros_like(input_tensor)
    output[:, :, 0::2, 1::2] = input_tensor[:, :, 0::2, 1::2]
    output[:, :, 1::2, 0::2] = input_tensor[:, :, 1::2, 0::2]
    return output


def checkerboard_nonanchor(input_tensor: Tensor) -> Tensor:
    output = torch.zeros_like(input_tensor)
    output[:, :, 0::2, 0::2] = input_tensor[:, :, 0::2, 0::2]
    output[:, :, 1::2, 1::2] = input_tensor[:, :, 1::2, 1::2]
    return output


def squeeze_anchor(input_tensor: Tensor) -> Tensor:
    batch_size, channels, height, width = input_tensor.shape
    output = input_tensor.new_zeros((batch_size, channels, height, width // 2))
    output[:, :, 0::2, :] = input_tensor[:, :, 0::2, 1::2]
    output[:, :, 1::2, :] = input_tensor[:, :, 1::2, 0::2]
    return output


def squeeze_nonanchor(input_tensor: Tensor) -> Tensor:
    batch_size, channels, height, width = input_tensor.shape
    output = input_tensor.new_zeros((batch_size, channels, height, width // 2))
    output[:, :, 0::2, :] = input_tensor[:, :, 0::2, 0::2]
    output[:, :, 1::2, :] = input_tensor[:, :, 1::2, 1::2]
    return output


def unsqueeze_anchor(input_tensor: Tensor) -> Tensor:
    batch_size, channels, height, width = input_tensor.shape
    output = input_tensor.new_zeros((batch_size, channels, height, width * 2))
    output[:, :, 0::2, 1::2] = input_tensor[:, :, 0::2, :]
    output[:, :, 1::2, 0::2] = input_tensor[:, :, 1::2, :]
    return output


def unsqueeze_nonanchor(input_tensor: Tensor) -> Tensor:
    batch_size, channels, height, width = input_tensor.shape
    output = input_tensor.new_zeros((batch_size, channels, height, width * 2))
    output[:, :, 0::2, 0::2] = input_tensor[:, :, 0::2, :]
    output[:, :, 1::2, 1::2] = input_tensor[:, :, 1::2, :]
    return output


def compress_symbols(
    gaussian_conditional: EntropyModel,
    input_tensor: Tensor,
    scales: Tensor,
    means: Tensor,
    squeeze_fn: Callable[[Tensor], Tensor],
    unsqueeze_fn: Callable[[Tensor], Tensor],
    symbols_list: List[int],
    indexes_list: List[int],
) -> Tensor:
    input_half = squeeze_fn(input_tensor)
    scales_half = squeeze_fn(scales)
    means_half = squeeze_fn(means)
    indexes = gaussian_conditional.build_indexes(scales_half)
    quantized = gaussian_conditional.quantize(input_half, "symbols", means_half)
    symbols_list.extend(quantized.reshape(-1).tolist())
    indexes_list.extend(indexes.reshape(-1).tolist())
    return unsqueeze_fn(quantized + means_half)


def decompress_symbols(
    gaussian_conditional: EntropyModel,
    scales: Tensor,
    means: Tensor,
    decoder: object,
    cdf: Sequence[Sequence[int]],
    cdf_lengths: Sequence[int],
    offsets: Sequence[int],
    squeeze_fn: Callable[[Tensor], Tensor],
    unsqueeze_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    scales_half = squeeze_fn(scales)
    means_half = squeeze_fn(means)
    indexes = gaussian_conditional.build_indexes(scales_half)
    decoded = decoder.decode_stream(
        indexes.reshape(-1).tolist(),
        cdf,
        cdf_lengths,
        offsets,
    )
    decoded_tensor = torch.tensor(
        decoded,
        device=scales.device,
        dtype=means_half.dtype,
    ).reshape(scales_half.shape)
    return unsqueeze_fn(decoded_tensor + means_half)
