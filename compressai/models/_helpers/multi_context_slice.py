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

"""MLIC-family per-slice entropy building blocks.

These parameter-holding modules (prior aggregation, entropy-parameter fusion,
intra-channel context wrappers, MLICv2 refinements) are the pieces the MLIC
family models wire together, ELIC-style, inside their ``__init__``. The
per-variant assembly itself lives in ``compressai.models.mlic`` so the model
module owns its entropy-stack layout (matching the TCM/STF/CCA convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.models._helpers.mlic import (
    ChannelContext,
    EntropyParameters,
    LinearGlobalInterContext,
    LinearGlobalIntraContext,
    VanillaGlobalInterContext,
    VanillaGlobalIntraContext,
)
from compressai.models._helpers.mlicv2 import (
    ContextReweighting,
    HGCPModule,
    RoPE2D,
)

__all__ = [
    "_MlicppSideLayout",
    "_MlicppPriorAggregation",
    "_MlicppEntropyParameters",
    "_MlicppIntraWrapper",
    "_MlicIntraWrapper",
    "_Mlicv2ContextRefinement",
    "_Mlicv2HgcpAnchorContext",
    "_build_lrp_input_builder",
    "_select_global_inter_factory",
]


def _select_num_heads(channels: int) -> int:
    target = max(1, channels // 32)
    while channels % target != 0:
        target -= 1
    return target


GlobalInterFactory = Callable[[int, int, int], nn.Module]


def _build_linear_global_inter_context(
    prior_ch: int,
    slice_ch: int,
    num_heads: int,
) -> nn.Module:
    return LinearGlobalInterContext(
        dim=prior_ch,
        out_dim=2 * slice_ch,
        num_heads=num_heads,
    )


def _build_vanilla_global_inter_context(
    prior_ch: int,
    slice_ch: int,
    num_heads: int,
) -> nn.Module:
    return VanillaGlobalInterContext(
        in_dim=prior_ch,
        out_dim=2 * slice_ch,
        num_heads=num_heads,
    )


@dataclass(frozen=True)
class _MlicppSideLayout:
    M: int
    slice_ch: int
    slice_index: int
    use_global_inter: bool = True

    @property
    def hyper_ch(self) -> int:
        return 2 * self.M

    @property
    def prior_ch(self) -> int:
        return self.slice_index * self.slice_ch

    @property
    def inter_ch(self) -> int:
        if self.slice_index and self.use_global_inter:
            return 2 * self.slice_ch
        return 0

    @property
    def channel_ch(self) -> int:
        return 4 * self.slice_ch if self.slice_index else 0

    @property
    def side_ch(self) -> int:
        return self.hyper_ch + self.prior_ch + self.inter_ch + self.channel_ch

    def split(self, side_params: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        hyper_end = self.hyper_ch
        prior_end = hyper_end + self.prior_ch
        inter_end = prior_end + self.inter_ch
        channel_end = inter_end + self.channel_ch
        return (
            side_params[:, :hyper_end],
            side_params[:, hyper_end:prior_end],
            side_params[:, prior_end:inter_end],
            side_params[:, inter_end:channel_end],
        )

    def hyper_means(self, side_params: Tensor) -> Tensor:
        hyper_params = side_params[:, : self.hyper_ch]
        return hyper_params[:, self.M :]

    def previous_slice(self, side_params: Tensor) -> Tensor:
        if self.prior_ch == 0:
            batch_size, _, height, width = side_params.shape
            return side_params.new_zeros(
                batch_size,
                self.slice_ch,
                height,
                width,
            )
        start = self.hyper_ch + self.prior_ch - self.slice_ch
        end = self.hyper_ch + self.prior_ch
        return side_params[:, start:end]


class _MlicppPriorAggregation(nn.Module):
    """Build the per-slice side layout consumed by MLIC-family leaf codecs."""

    def __init__(
        self,
        M: int,
        slice_ch: int,
        slice_index: int,
        global_inter_factory: Optional[
            GlobalInterFactory
        ] = _build_linear_global_inter_context,
    ) -> None:
        super().__init__()
        self.layout = _MlicppSideLayout(
            M=M,
            slice_ch=slice_ch,
            slice_index=slice_index,
            use_global_inter=global_inter_factory is not None,
        )
        if slice_index:
            prior_ch = self.layout.prior_ch
            self.channel_part = ChannelContext(in_dim=prior_ch, out_dim=slice_ch)
            self.global_inter_part = (
                global_inter_factory(
                    prior_ch,
                    slice_ch,
                    _select_num_heads(prior_ch),
                )
                if global_inter_factory is not None
                else None
            )
        else:
            self.channel_part = None
            self.global_inter_part = None

    def forward(self, params: Tensor) -> Tensor:
        hyper_params = params[:, : self.layout.hyper_ch]
        if self.layout.slice_index == 0:
            return hyper_params

        prior_y_hat = params[:, self.layout.hyper_ch :]
        if self.channel_part is None:
            raise RuntimeError("Expected prior aggregation modules for slice k > 0")
        parts = [hyper_params, prior_y_hat]
        if self.global_inter_part is not None:
            parts.append(self.global_inter_part(prior_y_hat))
        parts.append(self.channel_part(prior_y_hat))
        return torch.cat(parts, dim=1)


class _MlicppEntropyParameters(EntropyParameters):
    def __init__(
        self,
        layout: _MlicppSideLayout,
        *,
        step: str,
        anchor_context_ch: int = 0,
    ) -> None:
        if step == "anchor":
            in_dim = (
                anchor_context_ch
                + layout.hyper_ch
                + layout.inter_ch
                + layout.channel_ch
            )
        elif step == "non_anchor":
            in_dim = (
                2 * layout.slice_ch
                + layout.hyper_ch
                + layout.inter_ch
                + layout.channel_ch
            )
            if layout.slice_index:
                in_dim += 2 * layout.slice_ch
        else:
            raise ValueError(f'Invalid checkerboard step "{step}"')
        super().__init__(in_dim=in_dim, out_dim=2 * layout.slice_ch)
        self.layout = layout
        self.step = step
        self.anchor_context_ch = int(anchor_context_ch)

    def forward(self, params: Tensor) -> Tensor:
        if self.step == "anchor":
            anchor_ctx: Optional[Tensor] = None
            if self.anchor_context_ch:
                anchor_ctx = params[:, : self.anchor_context_ch]
                side_params = params[:, self.anchor_context_ch :]
            else:
                side_params = params
            local_ctx: Optional[Tensor] = None
            intra_ctx: Optional[Tensor] = None
        else:
            anchor_ctx = None
            local_ctx = params[:, : 2 * self.layout.slice_ch]
            side_start = 2 * self.layout.slice_ch
            side_end = side_start + self.layout.side_ch
            side_params = params[:, side_start:side_end]
            intra_ctx = params[:, side_end:]

        hyper_params, _, global_inter_ctx, channel_ctx = self.layout.split(side_params)
        if self.step == "anchor":
            parts = [hyper_params]
            if self.layout.slice_index:
                parts = [global_inter_ctx, channel_ctx, hyper_params]
            if anchor_ctx is not None:
                parts = [anchor_ctx, *parts]
        else:
            if local_ctx is None:
                raise RuntimeError("Expected local context for non-anchor step")
            parts = [local_ctx, hyper_params]
            if self.layout.slice_index:
                if intra_ctx is None:
                    raise RuntimeError("Expected intra context for slice k > 0")
                parts = [
                    local_ctx,
                    intra_ctx,
                    global_inter_ctx,
                    channel_ctx,
                    hyper_params,
                ]
        return super().forward(torch.cat(parts, dim=1))


class _MlicppIntraWrapper(LinearGlobalIntraContext):
    def __init__(self, layout: _MlicppSideLayout) -> None:
        super().__init__(dim=layout.slice_ch)
        self.layout = layout

    def forward(self, side_params: Tensor, anchor_y_hat: Tensor) -> Tensor:
        return super().forward(self.layout.previous_slice(side_params), anchor_y_hat)


class _MlicIntraWrapper(VanillaGlobalIntraContext):
    def __init__(self, layout: _MlicppSideLayout) -> None:
        super().__init__(dim=layout.slice_ch)
        self.layout = layout

    def forward(self, side_params: Tensor, anchor_y_hat: Tensor) -> Tensor:
        return super().forward(self.layout.previous_slice(side_params), anchor_y_hat)


class _Mlicv2ContextRefinement(nn.Module):
    def __init__(self, context: nn.Module, dim: int) -> None:
        super().__init__()
        self.context = context
        self.rope = RoPE2D(dim=dim)
        self.reweighting = ContextReweighting(dim=dim)

    def forward(self, *args: Tensor) -> Tensor:
        context = self.context(*args)
        context = self.rope(context)
        if not isinstance(context, Tensor):
            raise RuntimeError("Expected RoPE2D to return a tensor for one input")
        return self.reweighting(context)


class _Mlicv2HgcpAnchorContext(nn.Module):
    requires_side_params = True

    def __init__(self, M: int, slice_ch: int) -> None:
        super().__init__()
        self.hgcp = HGCPModule(
            M=M,
            slice_ch=slice_ch,
            num_heads=_select_num_heads(slice_ch),
        )

    def forward(self, _y_hat: Tensor, *, side_params: Tensor) -> Tensor:
        return self.hgcp(side_params)


def _build_lrp_input_builder(
    layout: _MlicppSideLayout,
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    def _lrp_inputs(side_params: Tensor, _params: Tensor, y_hat: Tensor) -> Tensor:
        _, prior_y_hat, _, _ = layout.split(side_params)
        return torch.cat([layout.hyper_means(side_params), prior_y_hat, y_hat], dim=1)

    return _lrp_inputs


def _build_mlicv2_global_inter_context(
    prior_ch: int,
    slice_ch: int,
    num_heads: int,
) -> nn.Module:
    return _Mlicv2ContextRefinement(
        _build_linear_global_inter_context(prior_ch, slice_ch, num_heads),
        dim=2 * slice_ch,
    )


def _select_global_inter_factory(
    variant: str,
) -> Optional[GlobalInterFactory]:
    """Return the per-slice global-inter context factory for ``variant``.

    ``None`` means the variant has no inter-slice global context (``mlic``).
    """
    if variant == "mlic+":
        return _build_vanilla_global_inter_context
    if variant == "mlicv2":
        return _build_mlicv2_global_inter_context
    if variant == "mlicpp":
        return _build_linear_global_inter_context
    if variant == "mlic":
        return None
    raise ValueError('variant must be one of "mlic", "mlic+", "mlicpp", or "mlicv2"')
