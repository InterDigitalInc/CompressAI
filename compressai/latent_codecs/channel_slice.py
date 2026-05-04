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

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec

__all__ = [
    "ChannelSliceLatentCodec",
]


@register_module("ChannelSliceLatentCodec")
class ChannelSliceLatentCodec(LatentCodec):
    """Channel-conditional entropy model with separate scale/mean heads and LRP.

    Splits ``y`` into equal-sized slices along the channel axis. For each
    slice ``k`` the previously decoded slices (truncated to
    ``max_support_slices``) are concatenated with ``latent_means`` /
    ``latent_scales`` and pushed through ``cc_mean_transforms[k]`` and
    ``cc_scale_transforms[k]`` to obtain ``mu`` / ``scale``. After the
    Gaussian conditional step, an optional latent residual prediction
    (LRP) head refines ``y_hat``.

    This is the channel-autoregressive entropy model from [Minnen2020]
    with the LRP refinement variant used in [Zhu2022] (STF / WACNN),
    [He2022] (ELIC) and many follow-up papers (MLIC++, TCM, ...).

    [Minnen2020]: `"Channel-wise Autoregressive Entropy Models for
    Learned Image Compression" <https://arxiv.org/abs/2007.08739>`_, by
    David Minnen and Saurabh Singh, ICIP 2020.

    [Zhu2022]: `"Transformer-based Transform Coding"
    <https://openreview.net/forum?id=IDwN6xjHnK8>`_, by Yinhao Zhu,
    Yang Yang and Taco Cohen, ICLR 2022.
    """

    cc_mean_transforms: nn.ModuleList
    cc_scale_transforms: nn.ModuleList
    lrp_transforms: nn.ModuleList
    gaussian_conditional: GaussianConditional

    def __init__(
        self,
        cc_mean_transforms: nn.ModuleList,
        cc_scale_transforms: nn.ModuleList,
        lrp_transforms: Optional[nn.ModuleList] = None,
        gaussian_conditional: Optional[GaussianConditional] = None,
        mean_support_transforms: Optional[nn.ModuleList] = None,
        scale_support_transforms: Optional[nn.ModuleList] = None,
        *,
        num_slices: Optional[int] = None,
        max_support_slices: int = -1,
        quantizer: str = "ste",
        lrp_scale: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._kwargs = kwargs

        inferred_num_slices = len(cc_mean_transforms)
        if num_slices is None:
            num_slices = inferred_num_slices
        if inferred_num_slices != num_slices:
            raise ValueError(
                "cc_mean_transforms must have num_slices entries "
                f"(got {inferred_num_slices}, expected {num_slices})"
            )
        if len(cc_scale_transforms) != num_slices:
            raise ValueError("cc_scale_transforms must have num_slices entries")
        if lrp_transforms is not None and len(lrp_transforms) != num_slices:
            raise ValueError("lrp_transforms must have num_slices entries")
        if mean_support_transforms is not None and len(mean_support_transforms) != num_slices:
            raise ValueError("mean_support_transforms must have num_slices entries")
        if scale_support_transforms is not None and len(scale_support_transforms) != num_slices:
            raise ValueError("scale_support_transforms must have num_slices entries")
        if quantizer not in ("ste", "noise"):
            raise ValueError(f"unknown quantizer {quantizer!r}")

        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)
        self.quantizer = quantizer
        self.lrp_scale = float(lrp_scale)
        self.cc_mean_transforms = cc_mean_transforms
        self.cc_scale_transforms = cc_scale_transforms
        self.mean_support_transforms = mean_support_transforms or nn.ModuleList(
            nn.Identity() for _ in range(num_slices)
        )
        self.scale_support_transforms = scale_support_transforms or nn.ModuleList(
            nn.Identity() for _ in range(num_slices)
        )
        self.lrp_transforms = lrp_transforms or nn.ModuleList(
            nn.Identity() for _ in range(num_slices)
        )
        self.gaussian_conditional = gaussian_conditional or GaussianConditional(None)

    def _support_slices(self, y_hat_slices: Sequence[Tensor]) -> List[Tensor]:
        if self.max_support_slices < 0:
            return list(y_hat_slices)
        return list(y_hat_slices[: self.max_support_slices])

    def _slice_params(
        self,
        slice_index: int,
        latent_means: Tensor,
        latent_scales: Tensor,
        y_hat_slices: Sequence[Tensor],
        spatial_shape: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        support = self._support_slices(y_hat_slices)
        mean_support = torch.cat([latent_means, *support], dim=1)
        mean_support = self.mean_support_transforms[slice_index](mean_support)
        mu = self.cc_mean_transforms[slice_index](mean_support)
        mu = mu[:, :, : spatial_shape[0], : spatial_shape[1]]
        scale_support = torch.cat([latent_scales, *support], dim=1)
        scale_support = self.scale_support_transforms[slice_index](scale_support)
        scale = self.cc_scale_transforms[slice_index](scale_support)
        scale = scale[:, :, : spatial_shape[0], : spatial_shape[1]]
        return mu, scale, mean_support

    def _apply_lrp(
        self, slice_index: int, mean_support: Tensor, y_hat_slice: Tensor
    ) -> Tensor:
        lrp = self.lrp_transforms[slice_index](
            torch.cat([mean_support, y_hat_slice], dim=1)
        )
        return y_hat_slice + self.lrp_scale * torch.tanh(lrp)

    def forward(
        self,
        y: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Dict[str, Any]:
        spatial_shape = (y.shape[2], y.shape[3])
        y_hat_slices: List[Tensor] = []
        y_likelihoods_slices: List[Tensor] = []

        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, dim=1)):
            mu, scale, mean_support = self._slice_params(
                slice_index, latent_means, latent_scales, y_hat_slices, spatial_shape
            )
            _, y_slice_likelihoods = self.gaussian_conditional(
                y_slice, scale, means=mu
            )
            if self.quantizer == "ste":
                y_hat_slice = quantize_ste(y_slice - mu) + mu
            else:
                y_hat_slice = self.gaussian_conditional.quantize(
                    y_slice, "noise" if self.training else "dequantize", mu
                )
            y_hat_slice = self._apply_lrp(slice_index, mean_support, y_hat_slice)
            y_hat_slices.append(y_hat_slice)
            y_likelihoods_slices.append(y_slice_likelihoods)

        return {
            "y_hat": torch.cat(y_hat_slices, dim=1),
            "likelihoods": {"y": torch.cat(y_likelihoods_slices, dim=1)},
        }

    def compress(
        self,
        y: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Dict[str, Any]:
        spatial_shape = (y.shape[2], y.shape[3])
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list: List[int] = []
        indexes_list: List[int] = []
        y_hat_slices: List[Tensor] = []

        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, dim=1)):
            mu, scale, mean_support = self._slice_params(
                slice_index, latent_means, latent_scales, y_hat_slices, spatial_shape
            )
            indexes = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu
            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(indexes.reshape(-1).tolist())
            y_hat_slice = self._apply_lrp(slice_index, mean_support, y_hat_slice)
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        return {
            "strings": [encoder.flush()],
            "shape": spatial_shape,
            "y_hat": torch.cat(y_hat_slices, dim=1),
        }

    def decompress(
        self,
        strings: Sequence[bytes],
        shape: Tuple[int, int],
        latent_means: Tensor,
        latent_scales: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0])
        y_hat_slices: List[Tensor] = []

        for slice_index in range(self.num_slices):
            mu, scale, mean_support = self._slice_params(
                slice_index, latent_means, latent_scales, y_hat_slices, shape
            )
            indexes = self.gaussian_conditional.build_indexes(scale)
            values = decoder.decode_stream(
                indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            y_q_slice = torch.tensor(
                values, device=mu.device, dtype=mu.dtype
            ).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(y_q_slice, mu)
            y_hat_slice = self._apply_lrp(slice_index, mean_support, y_hat_slice)
            y_hat_slices.append(y_hat_slice)

        return {"y_hat": torch.cat(y_hat_slices, dim=1)}
