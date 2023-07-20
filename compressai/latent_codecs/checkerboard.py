# Copyright (c) 2021-2022, InterDigital Communications, Inc
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

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyModel
from compressai.layers import CheckerboardMaskedConv2d
from compressai.ops import quantize_ste
from compressai.registry import register_module

from .base import LatentCodec
from .gaussian_conditional import GaussianConditionalLatentCodec

__all__ = [
    "CheckerboardLatentCodec",
]


@register_module("CheckerboardLatentCodec")
class CheckerboardLatentCodec(LatentCodec):
    """Reconstructs latent using 2-pass context model with checkerboard anchors.

    Checkerboard context model introduced in [He2021].

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    .. warning:: This implementation assumes that ``entropy_parameters``
       is a pointwise function, e.g., a composition of 1x1 convs and
       pointwise nonlinearities.

    .. note:: This implementation uses uniform noise for training quantization.

    .. code-block:: none

        0. Input:

        ■ ■ ■ ■
        ■ ■ ■ ■
        ■ ■ ■ ■

        1. Decode anchors:

        ◌ ■ ◌ ■
        ■ ◌ ■ ◌
        ◌ ■ ◌ ■

        2. Decode non-anchors:

        □ ◌ □ ◌
        ◌ □ ◌ □
        □ ◌ □ ◌

        3. End result:

        □ □ □ □
        □ □ □ □
        □ □ □ □

        LEGEND:
        □   decoded
        ◌   currently decoding
        ■   empty
    """

    latent_codec: Mapping[str, LatentCodec]

    entropy_parameters: nn.Module
    context_prediction: CheckerboardMaskedConv2d

    def __init__(
        self,
        latent_codec: Optional[Mapping[str, LatentCodec]] = None,
        entropy_parameters: Optional[nn.Module] = None,
        context_prediction: Optional[nn.Module] = None,
        forward_method="twopass",
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self.forward_method = forward_method
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.context_prediction = context_prediction or nn.Identity()
        self._set_group_defaults(
            "latent_codec",
            latent_codec,
            defaults={
                "y": lambda: GaussianConditionalLatentCodec(quantizer="ste"),
            },
            save_direct=True,
        )

    def forward(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        if self.forward_method == "twopass":
            return self._forward_twopass(y, side_params)
        y_hat = self.quantize(y)
        y_ctx = self._mask_anchor(self.context_prediction(y_hat))
        ctx_params = self.entropy_parameters(self.merge(side_params, y_ctx))
        y_out = self.latent_codec["y"](y, ctx_params)
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Do context prediction on STE-quantized y_hat instead."""
        y_hat_anchors = self._y_hat_anchors(y, side_params)
        y_ctx = self._mask_anchor(self.context_prediction(y_hat_anchors))
        ctx_params = self.entropy_parameters(self.merge(side_params, y_ctx))
        y_out = self.latent_codec["y"](y, ctx_params)
        # Reuse quantized y_hat that was used for non-anchor context prediction.
        y_hat = y_out["y_hat"]
        y_hat[..., 0::2, 0::2] = y_hat_anchors[..., 0::2, 0::2]
        y_hat[..., 1::2, 1::2] = y_hat_anchors[..., 1::2, 1::2]
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _y_hat_anchors(self, y, side_params):
        y_ctx = self.context_prediction(y).detach()
        y_ctx[:] = 0
        ctx_params = self.entropy_parameters(self.merge(side_params, y_ctx))
        ctx_params = self.latent_codec["y"].entropy_parameters(ctx_params)
        ctx_params = self._mask_non_anchor(ctx_params)  # Probably not needed.
        _, means_hat = ctx_params.chunk(2, 1)
        y_hat = quantize_ste(y - means_hat) + means_hat
        return y_hat

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        n, c, h, w = y.shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)
        y_ = self.unembed(y)
        y_strings_ = [None] * 2

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            ctx_params_i = self.entropy_parameters(self.merge(side_params_[i], y_ctx_i))
            y_out = self.latent_codec["y"].compress(y_[i], ctx_params_i)
            y_hat_[i] = y_out["y_hat"]
            [y_strings_[i]] = y_out["strings"]

        y_hat = self.embed(y_hat_)

        return {
            "strings": y_strings_,
            "shape": y_hat.shape[1:],
            "y_hat": y_hat,
        }

    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, ...], side_params: Tensor
    ) -> Dict[str, Any]:
        y_strings_ = strings
        n = len(y_strings_[0])
        assert len(y_strings_) == 2
        assert all(len(x) == n for x in y_strings_)

        c, h, w = shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            ctx_params_i = self.entropy_parameters(self.merge(side_params_[i], y_ctx_i))
            y_out = self.latent_codec["y"].decompress(
                [y_strings_[i]], shape=(h, w // 2), ctx_params=ctx_params_i
            )
            y_hat_[i] = y_out["y_hat"]

        y_hat = self.embed(y_hat_)

        return {
            "y_hat": y_hat,
        }

    def unembed(self, y: Tensor) -> Tensor:
        """Separate single tensor into two even/odd checkerboard chunks.

        .. code-block:: none

            □ ■ □ ■         □ □   ■ ■
            ■ □ ■ □   --->  □ □   ■ ■
            □ ■ □ ■         □ □   ■ ■
        """
        n, c, h, w = y.shape
        y_ = y.new_zeros((2, n, c, h, w // 2))
        y_[0, ..., 0::2, :] = y[..., 0::2, 0::2]
        y_[0, ..., 1::2, :] = y[..., 1::2, 1::2]
        y_[1, ..., 0::2, :] = y[..., 0::2, 1::2]
        y_[1, ..., 1::2, :] = y[..., 1::2, 0::2]
        return y_

    def embed(self, y_: Tensor) -> Tensor:
        """Combine two even/odd checkerboard chunks into single tensor.

        .. code-block:: none

            □ □   ■ ■         □ ■ □ ■
            □ □   ■ ■   --->  ■ □ ■ □
            □ □   ■ ■         □ ■ □ ■
        """
        num_chunks, n, c, h, w_half = y_.shape
        assert num_chunks == 2
        y = y_.new_zeros((n, c, h, w_half * 2))
        y[..., 0::2, 0::2] = y_[0, ..., 0::2, :]
        y[..., 1::2, 1::2] = y_[0, ..., 1::2, :]
        y[..., 0::2, 1::2] = y_[1, ..., 0::2, :]
        y[..., 1::2, 0::2] = y_[1, ..., 1::2, :]
        return y

    def _mask_anchor(self, y: Tensor) -> Tensor:
        y[..., 0::2, 0::2] = 0
        y[..., 1::2, 1::2] = 0
        return y

    def _mask_non_anchor(self, y: Tensor) -> Tensor:
        y[..., 0::2, 1::2] = 0
        y[..., 1::2, 0::2] = 0
        return y

    def merge(self, *args):
        return torch.cat(args, dim=1)

    def quantize(self, y: Tensor) -> Tensor:
        mode = "noise" if self.training else "dequantize"
        y_hat = EntropyModel.quantize(None, y, mode)
        return y_hat
