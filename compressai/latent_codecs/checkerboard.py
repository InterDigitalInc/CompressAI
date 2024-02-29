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

    See :py:class:`~compressai.models.sensetime.Cheng2020AnchorCheckerboard`
    for example usage.

    - `forward_method="onepass"` is fastest, but does not use
      quantization based on the intermediate means.
      Uses noise to model quantization.
    - `forward_method="twopass"` is slightly slower, but accurately
      quantizes via STE based on the intermediate means.
      Uses the same operations as [Chandelier2023].
    - `forward_method="twopass_faster"` uses slightly fewer
      redundant operations.

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    [Chandelier2023]: `"ELiC-ReImplemetation"
    <https://github.com/VincentChandelier/ELiC-ReImplemetation>`_, by
    Vincent Chandelier, 2023.

    .. warning:: This implementation assumes that ``entropy_parameters``
       is a pointwise function, e.g., a composition of 1x1 convs and
       pointwise nonlinearities.

    .. code-block:: none

        0. Input:

        □ □ □ □
        □ □ □ □
        □ □ □ □

        1. Decode anchors:

        ◌ □ ◌ □
        □ ◌ □ ◌
        ◌ □ ◌ □

        2. Decode non-anchors:

        ■ ◌ ■ ◌
        ◌ ■ ◌ ■
        ■ ◌ ■ ◌

        3. End result:

        ■ ■ ■ ■
        ■ ■ ■ ■
        ■ ■ ■ ■

        LEGEND:
        ■   decoded
        ◌   currently decoding
        □   empty
    """

    latent_codec: Mapping[str, LatentCodec]

    entropy_parameters: nn.Module
    context_prediction: CheckerboardMaskedConv2d

    def __init__(
        self,
        latent_codec: Optional[Mapping[str, LatentCodec]] = None,
        entropy_parameters: Optional[nn.Module] = None,
        context_prediction: Optional[nn.Module] = None,
        anchor_parity="even",
        forward_method="twopass",
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self.anchor_parity = anchor_parity
        self.non_anchor_parity = {"odd": "even", "even": "odd"}[anchor_parity]
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

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        if self.forward_method == "onepass":
            return self._forward_onepass(y, side_params)
        if self.forward_method == "twopass":
            return self._forward_twopass(y, side_params)
        if self.forward_method == "twopass_faster":
            return self._forward_twopass_faster(y, side_params)
        raise ValueError(f"Unknown forward method: {self.forward_method}")

    def _forward_onepass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Fast estimation with single pass of the entropy parameters network.

        It is faster than the twopass method (only one pass required!),
        but also less accurate.

        This method uses uniform noise to roughly model quantization.
        """
        y_hat = self.quantize(y)
        y_ctx = self._keep_only(self.context_prediction(y_hat), "non_anchor")
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec["y"](y, params)
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        The first pass gets ``y_hat`` and ``means_hat`` for the anchors.
        This ``y_hat`` is used as context to predict the non-anchors.
        The second pass gets ``y_hat`` for the non-anchors.
        The two ``y_hat`` tensors are then combined. The resulting
        ``y_hat`` models the effects of quantization more realistically.

        To compute ``y_hat_anchors``, we need the predicted ``means_hat``:
        ``y_hat = quantize_ste(y - means_hat) + means_hat``.
        Thus, two passes of ``entropy_parameters`` are necessary.

        """
        B, C, H, W = y.shape

        params = y.new_zeros((B, C * 2, H, W))

        y_hat_anchors = self._forward_twopass_step(
            y, side_params, params, self._y_ctx_zero(y), "anchor"
        )

        y_hat_non_anchors = self._forward_twopass_step(
            y, side_params, params, self.context_prediction(y_hat_anchors), "non_anchor"
        )

        y_hat = y_hat_anchors + y_hat_non_anchors
        y_out = self.latent_codec["y"](y, params)

        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass_step(
        self, y: Tensor, side_params: Tensor, params: Tensor, y_ctx: Tensor, step: str
    ) -> Dict[str, Any]:
        # NOTE: The _i variables contain only the current step's pixels.
        assert step in ("anchor", "non_anchor")

        params_i = self.entropy_parameters(self.merge(y_ctx, side_params))

        # Save params for current step. This is later used for entropy estimation.
        self._copy(params, params_i, step)

        # Apply latent_codec's "entropy_parameters()", if it exists. Usually identity.
        func = getattr(self.latent_codec["y"], "entropy_parameters", lambda x: x)
        params_i = func(params_i)

        # Keep only elements needed for current step.
        # It's not necessary to mask the rest out just yet, but it doesn't hurt.
        params_i = self._keep_only(params_i, step)
        y_i = self._keep_only(y, step)

        # Determine y_hat for current step, and mask out the other pixels.
        _, means_i = self.latent_codec["y"]._chunk(params_i)
        y_hat_i = self._keep_only(quantize_ste(y_i - means_i) + means_i, step)

        return y_hat_i

    def _forward_twopass_faster(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        This version was written based on the paper description.
        It is a tiny bit faster than the twopass method since
        it avoids a few redundant operations. The "probably unnecessary"
        operations can likely be removed as well.
        The speedup is very small, however.
        """
        y_ctx = self._y_ctx_zero(y)
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        func = getattr(self.latent_codec["y"], "entropy_parameters", lambda x: x)
        params = func(params)
        params = self._keep_only(params, "anchor")  # Probably unnecessary.
        _, means_hat = self.latent_codec["y"]._chunk(params)
        y_hat_anchors = quantize_ste(y - means_hat) + means_hat
        y_hat_anchors = self._keep_only(y_hat_anchors, "anchor")

        y_ctx = self.context_prediction(y_hat_anchors)
        y_ctx = self._keep_only(y_ctx, "non_anchor")  # Probably unnecessary.
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec["y"](y, params)

        # Reuse quantized y_hat that was used for non-anchor context prediction.
        y_hat = y_out["y_hat"]
        self._copy(y_hat, y_hat_anchors, "anchor")  # Probably unnecessary.

        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    @torch.no_grad()
    def _y_ctx_zero(self, y: Tensor) -> Tensor:
        """Create a zero tensor with correct shape for y_ctx."""
        y_ctx_meta = self.context_prediction(y.to("meta"))
        return y.new_zeros(y_ctx_meta.shape)

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        n, c, h, w = y.shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)
        y_ = self.unembed(y)
        y_strings_ = [None] * 2

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].compress(y_[i], params_i)
            y_hat_[i] = y_out["y_hat"]
            [y_strings_[i]] = y_out["strings"]

        y_hat = self.embed(y_hat_)

        return {
            "strings": y_strings_,
            "shape": y_hat.shape[1:],
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, ...],
        side_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        y_strings_ = strings
        n = len(y_strings_[0])
        assert len(y_strings_) == 2
        assert all(len(x) == n for x in y_strings_)

        c, h, w = shape
        y_i_shape = (h, w // 2)
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].decompress(
                [y_strings_[i]], y_i_shape, params_i
            )
            y_hat_[i] = y_out["y_hat"]

        y_hat = self.embed(y_hat_)

        return {
            "y_hat": y_hat,
        }

    def unembed(self, y: Tensor) -> Tensor:
        """Separate single tensor into two even/odd checkerboard chunks.

        .. code-block:: none

            ■ □ ■ □         ■ ■   □ □
            □ ■ □ ■   --->  ■ ■   □ □
            ■ □ ■ □         ■ ■   □ □
        """
        n, c, h, w = y.shape
        y_ = y.new_zeros((2, n, c, h, w // 2))
        if self.anchor_parity == "even":
            y_[0, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 1::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 0::2]
        else:
            y_[0, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 0::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 1::2]
        return y_

    def embed(self, y_: Tensor) -> Tensor:
        """Combine two even/odd checkerboard chunks into single tensor.

        .. code-block:: none

            ■ ■   □ □         ■ □ ■ □
            ■ ■   □ □   --->  □ ■ □ ■
            ■ ■   □ □         ■ □ ■ □
        """
        num_chunks, n, c, h, w_half = y_.shape
        assert num_chunks == 2
        y = y_.new_zeros((n, c, h, w_half * 2))
        if self.anchor_parity == "even":
            y[..., 0::2, 0::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 1::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[1, ..., 1::2, :]
        else:
            y[..., 0::2, 1::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 0::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[1, ..., 1::2, :]
        return y

    def _copy(self, dest: Tensor, src: Tensor, step: str) -> None:
        """Copy pixels in the current step."""
        assert step in ("anchor", "non_anchor")
        parity = self.anchor_parity if step == "anchor" else self.non_anchor_parity
        if parity == "even":
            dest[..., 0::2, 0::2] = src[..., 0::2, 0::2]
            dest[..., 1::2, 1::2] = src[..., 1::2, 1::2]
        else:
            dest[..., 0::2, 1::2] = src[..., 0::2, 1::2]
            dest[..., 1::2, 0::2] = src[..., 1::2, 0::2]

    def _keep_only(self, y: Tensor, step: str, inplace: bool = False) -> Tensor:
        """Keep only pixels in the current step, and zero out the rest."""
        return self._mask(
            y,
            parity=self.non_anchor_parity if step == "anchor" else self.anchor_parity,
            inplace=inplace,
        )

    def _mask(self, y: Tensor, parity: str, inplace: bool = False) -> Tensor:
        if not inplace:
            y = y.clone()
        if parity == "even":
            y[..., 0::2, 0::2] = 0
            y[..., 1::2, 1::2] = 0
        elif parity == "odd":
            y[..., 0::2, 1::2] = 0
            y[..., 1::2, 0::2] = 0
        elif parity == "all":
            y[:] = 0
        return y

    def merge(self, *args):
        return torch.cat(args, dim=1)

    def quantize(self, y: Tensor) -> Tensor:
        mode = "noise" if self.training else "dequantize"
        y_hat = EntropyModel.quantize(None, y, mode)
        return y_hat
