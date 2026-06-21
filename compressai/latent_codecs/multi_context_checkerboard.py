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

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.registry import register_module

from . import _checkerboard_helpers as _ckb
from . import _selective_checkerboard as _sel
from .base import LatentCodec
from .gaussian_conditional import GaussianConditionalLatentCodec

__all__ = ["MultiContextCheckerboardLatentCodec"]
LrpInputBuilder = Callable[[Tensor, Tensor, Tensor], Tensor]
LrpActivation = Optional[Callable[[Tensor], Tensor]]


@register_module("MultiContextCheckerboardLatentCodec")
class MultiContextCheckerboardLatentCodec(LatentCodec):
    """Two-pass checkerboard codec with separate heads and optional contexts.

    This is a sibling of :class:`CheckerboardLatentCodec` for models whose
    anchor and non-anchor passes use distinct entropy-parameter heads and
    optional per-pass latent residual prediction.

    Optional context hooks (``spatial_context_anchor`` /
    ``spatial_context_nonanchor`` / ``intra_channel_context_nonanchor``)
    are *omitted* from the entropy-parameters input when ``None``; they
    do not contribute zero-padding. The entropy-parameter heads must be
    sized to ``side_params.shape[1]`` plus the channel widths produced
    by whichever context modules are supplied for that pass.

    LRP modules are treated as raw residual predictors by default:
    ``lrp_activation`` (default: ``torch.tanh``) is applied before scaling.
    Set ``lrp_activation=None`` when the supplied LRP module already applies
    its own bounded activation.
    """

    def __init__(
        self,
        *,
        entropy_parameters_anchor: nn.Module,
        entropy_parameters_nonanchor: nn.Module,
        latent_codec: Optional[Mapping[str, LatentCodec]] = None,
        scale_table: Optional[List[float]] = None,
        gaussian_conditional: Optional[GaussianConditional] = None,
        spatial_context_anchor: Optional[nn.Module] = None,
        spatial_context_nonanchor: Optional[nn.Module] = None,
        intra_channel_context_nonanchor: Optional[nn.Module] = None,
        selective_predictor: Optional[nn.Module] = None,
        lrp_anchor: Optional[nn.Module] = None,
        lrp_nonanchor: Optional[nn.Module] = None,
        lrp_input_builder: Optional[LrpInputBuilder] = None,
        lrp_activation: LrpActivation = torch.tanh,
        lrp_scale: float = 0.5,
        anchor_parity: str = "even",
        quantizer: str = "ste",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if anchor_parity not in ("even", "odd"):
            raise ValueError(f'Invalid "anchor_parity" value "{anchor_parity}"')
        if quantizer != "ste":
            raise ValueError(f'Invalid quantizer "{quantizer}"')

        self._kwargs = kwargs
        self.anchor_parity = anchor_parity
        self.non_anchor_parity = {"odd": "even", "even": "odd"}[anchor_parity]
        self.quantizer = quantizer
        self.entropy_parameters_anchor = entropy_parameters_anchor
        self.entropy_parameters_nonanchor = entropy_parameters_nonanchor
        self.spatial_context_anchor = spatial_context_anchor
        self.spatial_context_nonanchor = spatial_context_nonanchor
        self.intra_channel_context_nonanchor = intra_channel_context_nonanchor
        self.selective_predictor = selective_predictor
        self.lrp_anchor = lrp_anchor
        self.lrp_nonanchor = lrp_nonanchor
        self.lrp_input_builder = lrp_input_builder
        self.lrp_activation = lrp_activation
        self.lrp_scale = float(lrp_scale)

        if latent_codec is None:
            latent_codec = {
                "y": GaussianConditionalLatentCodec(
                    scale_table=scale_table,
                    gaussian_conditional=gaussian_conditional,
                )
            }
        self.y = latent_codec["y"]
        self.latent_codec = latent_codec

    def forward(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        b, c, h, w = y.shape
        params = y.new_zeros((b, c * 2, h, w))
        y_hat_steps = []
        selective_masks = []

        for step in ("anchor", "non_anchor"):
            ctx_params = self._ctx_params(step, y, side_params, y_hat_steps)
            params_i = self._entropy_parameters(step)(ctx_params)
            params_i = _ckb.mask_all_but_step(
                params_i, step, anchor_parity=self.anchor_parity
            )
            _ckb.write_step(params, params_i, step, anchor_parity=self.anchor_parity)

            scales_i, means_i = self.y._chunk(params_i)
            selective_mask_i = _sel.selective_mask(
                self.selective_predictor,
                step,
                side_params,
                scales_i,
                means_i,
                anchor_parity=self.anchor_parity,
            )
            if selective_mask_i is not None:
                selective_masks.append(selective_mask_i)
            y_i = _ckb.mask_all_but_step(y, step, anchor_parity=self.anchor_parity)
            y_hat_i = self._quantize(y_i, means_i)
            y_hat_i = _ckb.mask_all_but_step(
                y_hat_i, step, anchor_parity=self.anchor_parity
            )
            y_hat_i = _sel.apply_selective_y_hat(
                step,
                y_hat_i,
                means_i,
                selective_mask_i,
                anchor_parity=self.anchor_parity,
            )
            lrp_input_y_hat = y_hat_i
            if step == "non_anchor":
                lrp_input_y_hat = y_hat_steps[0] + y_hat_i
            y_hat_i = self._apply_lrp(
                step,
                side_params,
                params_i,
                y_hat_i,
                lrp_input_y_hat,
            )
            y_hat_i = _sel.apply_selective_y_hat(
                step,
                y_hat_i,
                means_i,
                selective_mask_i,
                anchor_parity=self.anchor_parity,
            )
            y_hat_steps.append(y_hat_i)

        y_hat = y_hat_steps[0] + y_hat_steps[1]
        y_out = self.y(y, params)
        y_likelihoods = y_out["likelihoods"]["y"]
        if selective_masks:
            selective_mask = selective_masks[0] | selective_masks[1]
            y_likelihoods = torch.where(
                selective_mask, y_likelihoods, torch.ones_like(y_likelihoods)
            )

        return {
            "likelihoods": {
                "y": y_likelihoods,
            },
            "y_hat": y_hat,
        }

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        n, c, h, w = y.shape
        y_hat = y.new_zeros((n, c, h, w))
        y_hat_packed = y.new_zeros((2, n, c, h, w // 2))
        y_packed = _ckb.unembed(y, anchor_parity=self.anchor_parity)
        side_params_packed = _ckb.unembed(side_params, anchor_parity=self.anchor_parity)
        y_strings = [None] * 2

        for i, step in enumerate(("anchor", "non_anchor")):
            ctx_params_i = self._ctx_params_packed(
                i, step, side_params, side_params_packed, y_hat_packed
            )
            params_i = self._entropy_parameters(step)(ctx_params_i)
            scales_i, means_i = self.y._chunk(params_i)
            selective_mask_i = _sel.selective_mask_packed(
                self.selective_predictor,
                i,
                step,
                side_params,
                scales_i,
                means_i,
                anchor_parity=self.anchor_parity,
            )
            y_out = _sel.apply_selective_compression(
                self.y, y_packed[i], params_i, scales_i, means_i, selective_mask_i
            )
            y_hat_for_lrp = y_hat_packed.clone()
            y_hat_for_lrp[i] = y_out["y_hat"]
            y_hat_i = self._apply_lrp_packed(
                i, step, side_params, params_i, y_hat_for_lrp
            )
            y_hat_i = _sel.apply_selective_y_hat_packed(
                y_hat_i, means_i, selective_mask_i
            )
            y_hat_packed[i] = y_hat_i
            [y_strings[i]] = y_out["strings"]

        y_hat[:] = _ckb.embed(y_hat_packed, anchor_parity=self.anchor_parity)

        return {
            "strings": y_strings,
            "shape": y_hat.shape[1:],
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, ...],
        side_params: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        y_strings = strings
        n = len(y_strings[0])
        assert len(y_strings) == 2
        assert all(len(x) == n for x in y_strings)

        c, h, w = shape
        y_i_shape = (c, h, w // 2)
        y_hat_packed = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_packed = _ckb.unembed(side_params, anchor_parity=self.anchor_parity)

        for i, step in enumerate(("anchor", "non_anchor")):
            ctx_params_i = self._ctx_params_packed(
                i, step, side_params, side_params_packed, y_hat_packed
            )
            params_i = self._entropy_parameters(step)(ctx_params_i)
            scales_i, means_i = self.y._chunk(params_i)
            selective_mask_i = _sel.selective_mask_packed(
                self.selective_predictor,
                i,
                step,
                side_params,
                scales_i,
                means_i,
                anchor_parity=self.anchor_parity,
            )
            y_out = _sel.apply_selective_decompression(
                self.y,
                y_strings[i],
                y_i_shape,
                params_i,
                scales_i,
                means_i,
                selective_mask_i,
            )
            y_hat_for_lrp = y_hat_packed.clone()
            y_hat_for_lrp[i] = y_out["y_hat"]
            y_hat_i = self._apply_lrp_packed(
                i, step, side_params, params_i, y_hat_for_lrp
            )
            y_hat_packed[i] = _sel.apply_selective_y_hat_packed(
                y_hat_i, means_i, selective_mask_i
            )

        return {
            "y_hat": _ckb.embed(y_hat_packed, anchor_parity=self.anchor_parity),
        }

    def _ctx_params(
        self,
        step: str,
        y: Tensor,
        side_params: Tensor,
        y_hat_steps: List[Tensor],
    ) -> Tensor:
        ctx_parts: List[Tensor] = []
        spatial = self._spatial_context_module(step)
        if spatial is not None:
            y_hat_for_ctx = _ckb.mask_all(y) if step == "anchor" else y_hat_steps[0]
            ctx_parts.append(
                self._apply_spatial_context(spatial, y_hat_for_ctx, side_params)
            )
        ctx_parts.append(side_params)
        if step == "non_anchor" and self.intra_channel_context_nonanchor is not None:
            ctx_parts.append(
                self.intra_channel_context_nonanchor(side_params, y_hat_steps[0])
            )
        return _ckb.merge(*ctx_parts)

    def _ctx_params_packed(
        self,
        step_index: int,
        step: str,
        side_params: Tensor,
        side_params_packed: Tensor,
        y_hat_packed: Tensor,
    ) -> Tensor:
        ctx_parts: List[Tensor] = []
        spatial = self._spatial_context_module(step)
        if spatial is not None:
            y_hat_full = _ckb.embed(y_hat_packed, anchor_parity=self.anchor_parity)
            if step == "anchor":
                y_hat_full = _ckb.mask_all(y_hat_full)
            y_ctx = self._apply_spatial_context(spatial, y_hat_full, side_params)
            ctx_parts.append(
                _ckb.unembed(y_ctx, anchor_parity=self.anchor_parity)[step_index]
            )
        ctx_parts.append(side_params_packed[step_index])
        if step == "non_anchor" and self.intra_channel_context_nonanchor is not None:
            y_hat_full = _ckb.embed(y_hat_packed, anchor_parity=self.anchor_parity)
            intra_ctx = self.intra_channel_context_nonanchor(side_params, y_hat_full)
            ctx_parts.append(
                _ckb.unembed(intra_ctx, anchor_parity=self.anchor_parity)[step_index]
            )
        return _ckb.merge(*ctx_parts)

    def _spatial_context_module(self, step: str) -> Optional[nn.Module]:
        if step == "anchor":
            return self.spatial_context_anchor
        return self.spatial_context_nonanchor

    def _apply_spatial_context(
        self,
        spatial: nn.Module,
        y_hat: Tensor,
        side_params: Tensor,
    ) -> Tensor:
        if getattr(spatial, "requires_side_params", False):
            return spatial(y_hat, side_params=side_params)
        return spatial(y_hat)

    def _entropy_parameters(self, step: str) -> nn.Module:
        if step == "anchor":
            return self.entropy_parameters_anchor
        return self.entropy_parameters_nonanchor

    def _quantize(self, y: Tensor, means: Tensor) -> Tensor:
        return quantize_ste(y - means) + means

    def _apply_lrp(
        self,
        step: str,
        side_params: Tensor,
        params: Tensor,
        y_hat: Tensor,
        lrp_input_y_hat: Optional[Tensor] = None,
    ) -> Tensor:
        lrp = self.lrp_anchor if step == "anchor" else self.lrp_nonanchor
        if lrp is None:
            return y_hat
        if lrp_input_y_hat is None:
            lrp_input_y_hat = y_hat
        lrp_input = self._build_lrp_input(side_params, params, lrp_input_y_hat)
        y_hat = y_hat + self.lrp_scale * self._activate_lrp(lrp(lrp_input))
        return _ckb.mask_all_but_step(y_hat, step, anchor_parity=self.anchor_parity)

    def _apply_lrp_packed(
        self,
        step_index: int,
        step: str,
        side_params: Tensor,
        params: Tensor,
        y_hat_packed: Tensor,
    ) -> Tensor:
        lrp = self.lrp_anchor if step == "anchor" else self.lrp_nonanchor
        if lrp is None:
            return y_hat_packed[step_index]
        y_hat = _ckb.embed(y_hat_packed, anchor_parity=self.anchor_parity)
        params_full = _ckb.embed_step(
            step_index,
            params,
            side_params.shape[-1],
            anchor_parity=self.anchor_parity,
        )
        lrp_input = self._build_lrp_input(side_params, params_full, y_hat)
        lrp_out = _ckb.unembed(lrp(lrp_input), anchor_parity=self.anchor_parity)[
            step_index
        ]
        return y_hat_packed[step_index] + self.lrp_scale * self._activate_lrp(lrp_out)

    def _build_lrp_input(
        self, side_params: Tensor, params: Tensor, y_hat: Tensor
    ) -> Tensor:
        if self.lrp_input_builder is not None:
            return self.lrp_input_builder(side_params, params, y_hat)
        return _ckb.merge(side_params, y_hat)

    def _activate_lrp(self, residual: Tensor) -> Tensor:
        if self.lrp_activation is None:
            return residual
        return self.lrp_activation(residual)
