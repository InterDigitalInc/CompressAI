# Copyright (c) 2021-2024, InterDigital Communications, Inc
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

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec

__all__ = [
    "GaussianConditionalLatentCodec",
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
        chunks: Tuple[str] = ("scales", "means"),
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
