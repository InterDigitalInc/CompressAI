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

"""Mean / scale split channel-context heads for Family 1 models.

The :class:`MeanScaleContextHead` keeps a separate ``mean_cc`` and
``scale_cc`` Sequential — matching the historical ``cc_mean_transforms`` /
``cc_scale_transforms`` ModuleList layout used by STF / WACNN / TCM /
CCA — and concatenates their outputs to form the
``channel_context.y{k}`` entry expected by
:class:`~compressai.latent_codecs.ChannelGroupsLatentCodec`.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from torch import Tensor

from compressai.latent_codecs._slice_helpers import make_entropy_transform

__all__ = [
    "MeanScaleContextHead",
    "build_mean_scale_head",
]


class MeanScaleContextHead(nn.Module):
    """Channel-context head with separate mean / scale sub-networks.

    Internal layout::

        mean_cc:  in_channels -> ... -> slice_ch
        scale_cc: in_channels -> ... -> slice_ch

    Forward output is ``cat([scale_cc(...), mean_cc(...)], dim=1)`` of shape
    ``(B, 2 * slice_ch, H, W)`` — order matches
    :class:`GaussianConditionalLatentCodec` ``chunks=("scales", "means")``.
    Optional ``mean_support_transform`` / ``scale_support_transform`` run
    independently on the input before the sub-networks (used for SWAtten in
    TCM and NAFTransform in CCA).

    When ``side_split > 0`` the head expects its input to be the
    concatenation ``cat(latent_means(side_split), latent_scales(side_split),
    *prev_y_hat)`` produced by
    :class:`~compressai.latent_codecs.ChannelGroupsLatentCodec` running in
    ``side_in_context=True`` mode. The head splits the leading
    ``2 * side_split`` channels back into ``latent_means`` /
    ``latent_scales`` and routes:

    - ``mean_cc(cat(latent_means, *prev_y_hat))``
    - ``scale_cc(cat(latent_scales, *prev_y_hat))``

    so each sub-network sees the same input shape it would have under the
    pre-refactor STF / WACNN / TCM / CCA wiring (``cc_mean_transforms[k]`` /
    ``cc_scale_transforms[k]``). This keeps state-dict weights compatible
    with the legacy layout when migrating via
    ``convert_*_checkpoint.py``.

    When ``side_split == 0`` (default) the head is generic: ``mean_cc`` and
    ``scale_cc`` both see the full input, no internal split.

    When ``emit_mean_support=True`` (only meaningful with ``side_split > 0``)
    the head appends the ``mean_in = cat(latent_means, *prev_y_hat)`` tensor
    to the output, producing
    ``cat(scale, mean, mean_in)`` of shape
    ``(B, 2*slice_ch + side_split + sum(prev_groups), H, W)``. This trailing
    block is consumed by :class:`LRPGaussianLatentCodec` (with matching
    ``mean_support_trail_channels``) to recover the upstream STF / WACNN
    LRP input layout (``cat(latent_means, *prev_y_hat, y_hat)``), enabling
    byte-for-byte transfer of upstream LRP weights.
    """

    mean_cc: nn.Module
    scale_cc: nn.Module
    mean_support_transform: nn.Module
    scale_support_transform: nn.Module

    def __init__(
        self,
        mean_cc: nn.Module,
        scale_cc: nn.Module,
        mean_support_transform: Optional[nn.Module] = None,
        scale_support_transform: Optional[nn.Module] = None,
        *,
        side_split: int = 0,
        emit_mean_support: bool = False,
    ) -> None:
        super().__init__()
        self.mean_cc = mean_cc
        self.scale_cc = scale_cc
        self.mean_support_transform = mean_support_transform or nn.Identity()
        self.scale_support_transform = scale_support_transform or nn.Identity()
        self.side_split = int(side_split)
        self.emit_mean_support = bool(emit_mean_support)
        if self.emit_mean_support and self.side_split <= 0:
            raise ValueError(
                "emit_mean_support=True requires side_split > 0 to recover "
                "the legacy mean_support layout cat(latent_means, *prev_y_hat)."
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.side_split > 0:
            split = self.side_split
            latent_means = x[:, :split]
            latent_scales = x[:, split : 2 * split]
            prev_y_hat = x[:, 2 * split :]
            mean_in = torch.cat([latent_means, prev_y_hat], dim=1)
            scale_in = torch.cat([latent_scales, prev_y_hat], dim=1)
        else:
            mean_in = scale_in = x
        mean = self.mean_cc(self.mean_support_transform(mean_in))
        scale = self.scale_cc(self.scale_support_transform(scale_in))
        out = torch.cat([scale, mean], dim=1)
        if self.emit_mean_support:
            out = torch.cat([out, mean_in], dim=1)
        return out


def build_mean_scale_head(
    slice_ch: int,
    support_ch: int,
    *,
    widths: Sequence[int] = (224, 128),
    support_transform_factory: Optional[Callable[[int, int], nn.Module]] = None,
    side_split: int = 0,
    emit_mean_support: bool = False,
) -> MeanScaleContextHead:
    """Construct a :class:`MeanScaleContextHead` with default conv-stack heads.

    Parameters
    ----------
    slice_ch
        Channel count of the slice being predicted (per-sub-head output).
    support_ch
        FULL input channel count to the head (i.e., what
        :class:`ChannelGroupsLatentCodec` will hand it). When
        ``side_split > 0`` this equals ``2 * side_split + slice_ch *
        support_count``; the head will internally split off ``2 * side_split``
        channels and route ``side_split`` each to ``mean_cc`` / ``scale_cc``,
        so each sub-network receives ``support_ch - side_split`` channels.
        When ``side_split == 0`` ``mean_cc`` / ``scale_cc`` see the full
        ``support_ch`` directly.
    widths
        Hidden conv widths inside the ``mean_cc`` / ``scale_cc`` Sequentials.
        STF / WACNN use ``(224, 176, 128, 64)``; TCM / CCA use
        ``(224, 128)``.
    support_transform_factory
        ``(in_ch, out_ch) -> nn.Module``. When supplied, builds independent
        instances for the mean and scale paths (e.g., per-slice SWAtten in
        TCM or NAFTransform in CCA). Both transforms are expected to
        preserve channel count and are applied to the per-path input
        (``support_ch - side_split`` channels).
    side_split
        Number of leading channels in the input that hold ``latent_means``
        (with ``latent_scales`` immediately after, also ``side_split`` wide).
        Set to the hyper-synthesis output channel count ``M`` for the
        Family 1 ``side_in_context=True`` wiring; leave ``0`` for generic
        usage.
    emit_mean_support
        Forwarded to :class:`MeanScaleContextHead`. Why this flag exists:
        the upstream STF / WACNN / TCM / CCA LRP transform consumes
        ``cat(latent_means, *prev_y_hat, y_hat)`` (i.e. ``M + slice_ch *
        (support_count + 1)`` channels — variable per slice). The Phase 3
        leaf only sees the channel-context ``ctx_params`` (= 2*slice_ch) and
        ``y_hat``, which would force an architectural change to the LRP
        transform input width and prevent byte-for-byte transfer of upstream
        LRP weights. Setting ``emit_mean_support=True`` makes the head
        append ``mean_in = cat(latent_means, *prev_y_hat)`` to its output;
        :class:`LRPGaussianLatentCodec` (with matching
        ``mean_support_trail_channels``) then strips that trailing block off
        ``ctx_params``, feeds only the leading ``2*slice_ch`` to the
        Gaussian conditional's ``chunks=("scales","means")`` step, and uses
        the trailing block as the LRP input — recovering the upstream layout
        exactly.
    """
    sub_in_ch = support_ch - side_split
    mean_cc = make_entropy_transform(sub_in_ch, slice_ch, widths=widths)
    scale_cc = make_entropy_transform(sub_in_ch, slice_ch, widths=widths)
    mean_support: Optional[nn.Module]
    scale_support: Optional[nn.Module]
    if support_transform_factory is not None:
        mean_support = support_transform_factory(sub_in_ch, sub_in_ch)
        scale_support = support_transform_factory(sub_in_ch, sub_in_ch)
    else:
        mean_support = None
        scale_support = None
    return MeanScaleContextHead(
        mean_cc=mean_cc,
        scale_cc=scale_cc,
        mean_support_transform=mean_support,
        scale_support_transform=scale_support,
        side_split=side_split,
        emit_mean_support=emit_mean_support,
    )
