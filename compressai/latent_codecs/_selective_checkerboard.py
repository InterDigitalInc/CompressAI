# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

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

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GaussianConditional

from . import checkerboard as _ckb

__all__ = [
    "apply_selective_y_hat",
    "apply_selective_y_hat_packed",
    "apply_selective_compression",
    "apply_selective_decompression",
    "compress_selected",
    "decompress_selected",
    "selective_mask",
    "selective_mask_packed",
]


def selective_mask(
    selective_predictor: Optional[nn.Module],
    step: str,
    side_params: Tensor,
    scales: Tensor,
    means: Tensor,
    *,
    anchor_parity: str,
) -> Optional[Tensor]:
    if selective_predictor is None:
        return None
    selective_map = selective_predictor(
        side_params=side_params,
        scales=scales,
        means=means,
        step=step,
    )
    if isinstance(selective_map, dict):
        selective_map = selective_map["selective_map"]
    if selective_map.shape != scales.shape:
        selective_map = selective_map.expand_as(scales)
    if selective_map.dtype == torch.bool:
        mask = selective_map
    else:
        mask = selective_map >= 0.5
    return _ckb.mask_all_but_step(mask, step, anchor_parity=anchor_parity)


def selective_mask_packed(
    selective_predictor: Optional[nn.Module],
    step_index: int,
    step: str,
    side_params: Tensor,
    scales: Tensor,
    means: Tensor,
    *,
    anchor_parity: str,
) -> Optional[Tensor]:
    if selective_predictor is None:
        return None
    width = side_params.shape[-1]
    scales_full = _ckb.embed_step(
        step_index, scales, width, anchor_parity=anchor_parity
    )
    means_full = _ckb.embed_step(step_index, means, width, anchor_parity=anchor_parity)
    mask = selective_mask(
        selective_predictor,
        step,
        side_params,
        scales_full,
        means_full,
        anchor_parity=anchor_parity,
    )
    if mask is None:
        return None
    return _ckb.unembed(mask, anchor_parity=anchor_parity)[step_index]


def apply_selective_y_hat(
    step: str,
    y_hat: Tensor,
    means: Tensor,
    selective_mask: Optional[Tensor],
    *,
    anchor_parity: str,
) -> Tensor:
    if selective_mask is None:
        return y_hat
    y_hat = torch.where(selective_mask, y_hat, means)
    return _ckb.mask_all_but_step(y_hat, step, anchor_parity=anchor_parity)


def apply_selective_y_hat_packed(
    y_hat: Tensor,
    means: Tensor,
    selective_mask: Optional[Tensor],
) -> Tensor:
    if selective_mask is None:
        return y_hat
    return torch.where(selective_mask, y_hat, means)


def apply_selective_compression(
    latent_codec: Any,
    y: Tensor,
    params: Tensor,
    scales: Tensor,
    means: Tensor,
    selective_mask: Optional[Tensor],
) -> Dict[str, Any]:
    if selective_mask is None:
        return latent_codec.compress(y, params)
    return compress_selected(
        latent_codec.gaussian_conditional, y, scales, means, selective_mask
    )


def apply_selective_decompression(
    latent_codec: Any,
    strings: List[bytes],
    shape: Tuple[int, ...],
    params: Tensor,
    scales: Tensor,
    means: Tensor,
    selective_mask: Optional[Tensor],
) -> Dict[str, Any]:
    if selective_mask is None:
        return latent_codec.decompress([strings], shape, params)
    y_hat = decompress_selected(
        latent_codec.gaussian_conditional, strings, scales, means, selective_mask
    )
    assert y_hat.shape[1:] == shape
    return {"y_hat": y_hat}


def compress_selected(
    gaussian_conditional: GaussianConditional,
    y: Tensor,
    scales: Tensor,
    means: Tensor,
    selective_mask: Tensor,
) -> Dict[str, Any]:
    indexes = gaussian_conditional.build_indexes(scales)
    y_strings = []
    y_hat = means.clone()

    for sample_index in range(y.shape[0]):
        mask = selective_mask[sample_index].reshape(-1)
        if not mask.any():
            y_strings.append(b"")
            continue

        y_i = y[sample_index].reshape(-1)[mask].unsqueeze(0)
        indexes_i = indexes[sample_index].reshape(-1)[mask].unsqueeze(0)
        means_i = means[sample_index].reshape(-1)[mask].unsqueeze(0)
        y_string = gaussian_conditional.compress(y_i, indexes_i, means_i)[0]
        y_hat_i = gaussian_conditional.decompress([y_string], indexes_i, means=means_i)
        y_hat[sample_index].reshape(-1)[mask] = y_hat_i.reshape(-1).to(y_hat.dtype)
        y_strings.append(y_string)

    return {"strings": [y_strings], "shape": y.shape[2:4], "y_hat": y_hat}


def decompress_selected(
    gaussian_conditional: GaussianConditional,
    strings: List[bytes],
    scales: Tensor,
    means: Tensor,
    selective_mask: Tensor,
) -> Tensor:
    indexes = gaussian_conditional.build_indexes(scales)
    y_hat = means.clone()

    for sample_index, y_string in enumerate(strings):
        mask = selective_mask[sample_index].reshape(-1)
        if not mask.any():
            continue
        indexes_i = indexes[sample_index].reshape(-1)[mask].unsqueeze(0)
        means_i = means[sample_index].reshape(-1)[mask].unsqueeze(0)
        y_hat_i = gaussian_conditional.decompress([y_string], indexes_i, means=means_i)
        y_hat[sample_index].reshape(-1)[mask] = y_hat_i.reshape(-1).to(y_hat.dtype)

    return y_hat
