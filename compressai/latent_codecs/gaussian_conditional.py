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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec

__all__ = [
    "GaussianConditionalLatentCodec",
    "LRPGaussianLatentCodec",
]


@register_module("GaussianConditionalLatentCodec")
class GaussianConditionalLatentCodec(LatentCodec):
    """Gaussian conditional for compressing latent ``y`` using ``ctx_params``.

    Probability model for Gaussian of ``(scales, means)``.

    Gaussian conditonal entropy model introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. note:: Unlike the original paper, which models only the scale
       (i.e. "width") of the Gaussian, this implementation models both
       the scale and the mean (i.e. "center") of the Gaussian.

    .. code-block:: none

                          ctx_params
                              │
                              ▼
                              │
                           ┌──┴──┐
                           │  EP │
                           └──┬──┘
                              │
               ┌───┐  y_hat   ▼
        y ──►──┤ Q ├────►────····──►── y_hat
               └───┘          GC

    """

    gaussian_conditional: GaussianConditional
    entropy_parameters: nn.Module

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]] = None,
        gaussian_conditional: Optional[GaussianConditional] = None,
        entropy_parameters: Optional[nn.Module] = None,
        quantizer: str = "noise",
        chunks: Tuple[str, ...] = ("scales", "means"),
        **kwargs,
    ):
        super().__init__()
        self.quantizer = quantizer
        self.gaussian_conditional = gaussian_conditional or GaussianConditional(
            scale_table, **kwargs
        )
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.chunks = tuple(chunks)

    def forward(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        if self.quantizer == "ste":
            y_hat = quantize_ste(y - means_hat) + means_hat
        return {"likelihoods": {"y": y_likelihoods}, "y_hat": y_hat}

    def compress(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means_hat)
        y_hat = self.gaussian_conditional.decompress(
            y_strings, indexes, means=means_hat
        )
        return {"strings": [y_strings], "shape": y.shape[2:4], "y_hat": y_hat}

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, int],
        ctx_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        (y_strings,) = strings
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            y_strings, indexes, means=means_hat
        )
        assert y_hat.shape[2:4] == shape
        return {"y_hat": y_hat}

    def _chunk(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        scales, means = None, None
        if self.chunks == ("scales",):
            scales = params
        if self.chunks == ("means",):
            means = params
        if self.chunks == ("scales", "means"):
            scales, means = params.chunk(2, 1)
        if self.chunks == ("means", "scales"):
            means, scales = params.chunk(2, 1)
        return scales, means


@register_module("LRPGaussianLatentCodec")
class LRPGaussianLatentCodec(GaussianConditionalLatentCodec):
    """Gaussian conditional with a latent residual prediction (LRP) refinement.

    Wraps :class:`GaussianConditionalLatentCodec` and applies an additive LRP
    head to the quantized latent ``y_hat``. The LRP head receives
    ``cat(mean_support, y_hat)`` and produces a residual scaled by
    ``lrp_scale`` and squashed by ``tanh``::

        y_hat = y_hat + lrp_scale * tanh(lrp_transform(cat(mean_support, y_hat)))

    Used as the per-slice leaf for Family 1 channel-slice models (STF / WACNN
    / TCM / CCA-main, plus the first ``K-2`` slices of CCA-aux). The LRP
    refinement variant was introduced in [Zhu2022] and is widely adopted by
    follow-up work (ELIC checkerboard variants, MLIC++, TCM, ...).

    The ``mean_support`` tensor that feeds the LRP head depends on
    ``mean_support_trail_channels``:

    - ``0`` (default): ``mean_support = ctx_params`` — LRP sees the full
      ctx_params concatenated with ``y_hat``.
    - ``> 0``: ``ctx_params`` is expected to be laid out as
      ``cat(gaussian_params, mean_support)`` where the trailing
      ``mean_support_trail_channels`` block carries
      ``cat(latent_means, *prev_y_hat)``. The leaf forwards only
      ``gaussian_params = ctx_params[:, :-mean_support_trail_channels]`` to
      the underlying :class:`GaussianConditionalLatentCodec` (so chunk
      semantics are preserved) and uses the trailing block as
      ``mean_support`` for LRP. This recovers the upstream STF / WACNN LRP
      input layout (``cat(latent_means, *prev_y_hat, y_hat)``), enabling
      byte-for-byte transfer of upstream LRP weights.

    [Zhu2022]: `"Transformer-based Transform Coding"
    <https://openreview.net/forum?id=IDwN6xjHnK8>`_, by Yinhao Zhu, Yang Yang
    and Taco Cohen, ICLR 2022.
    """

    lrp_transform: nn.Module

    def __init__(
        self,
        lrp_transform: nn.Module,
        *,
        lrp_scale: float = 0.5,
        mean_support_trail_channels: int = 0,
        **gc_kwargs: Any,
    ) -> None:
        super().__init__(**gc_kwargs)
        self.lrp_transform = lrp_transform
        self.lrp_scale = float(lrp_scale)
        self.mean_support_trail_channels = int(mean_support_trail_channels)

    def _split_ctx_params(self, ctx_params: Tensor) -> Tuple[Tensor, Tensor]:
        if self.mean_support_trail_channels <= 0:
            return ctx_params, ctx_params
        trail = self.mean_support_trail_channels
        gaussian_params = ctx_params[:, :-trail]
        mean_support = ctx_params[:, -trail:]
        return gaussian_params, mean_support

    def _apply_lrp(self, mean_support: Tensor, y_hat: Tensor) -> Tensor:
        lrp = self.lrp_scale * torch.tanh(
            self.lrp_transform(torch.cat([mean_support, y_hat], dim=1))
        )
        return y_hat + lrp

    def forward(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params, mean_support = self._split_ctx_params(ctx_params)
        out = super().forward(y, gaussian_params)
        out["y_hat"] = self._apply_lrp(mean_support, out["y_hat"])
        return out

    def compress(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params, mean_support = self._split_ctx_params(ctx_params)
        out = super().compress(y, gaussian_params)
        out["y_hat"] = self._apply_lrp(mean_support, out["y_hat"])
        return out

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, int],
        ctx_params: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        gaussian_params, mean_support = self._split_ctx_params(ctx_params)
        out = super().decompress(strings, shape, gaussian_params, **kwargs)
        out["y_hat"] = self._apply_lrp(mean_support, out["y_hat"])
        return out
