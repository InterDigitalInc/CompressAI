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

"""Mean / scale split channel-context heads for model-specific entropy stacks.

The :class:`MeanScaleContextHead` keeps a separate ``mean_cc`` and
``scale_cc`` Sequential — matching the historical ``cc_mean_transforms`` /
``cc_scale_transforms`` ModuleList layout used by STF / WACNN / TCM /
CCA — and concatenates their outputs to form the
``channel_context.y{k}`` entry used by those models.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import torch
import torch.nn as nn

from torch import Tensor

__all__ = [
    "MeanScaleContextHead",
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
    *prev_y_hat)`` produced by the side-parameter channel-groups path. The
    head splits the leading
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

    When ``emit_mean_support`` is truthy (only meaningful with
    ``side_split > 0``) the head appends a copy of the mean-path tensor to
    the output, producing
    ``cat(scale, mean, mean_support)`` of shape
    ``(B, 2*slice_ch + side_split + sum(prev_groups), H, W)``. Two flavours:

    - ``"pre"`` (legacy ``True``) — emit the raw ``mean_in =
      cat(latent_means, *prev_y_hat)`` (i.e., before
      ``mean_support_transform``). STF / WACNN / TCM use this because their
      ``mean_support_transform`` is :class:`Identity` (or the upstream LRP
      input is the un-transformed mean_in).
    - ``"post"`` — emit ``mean_support_transform(mean_in)`` (the same tensor
      that feeds ``mean_cc``). CCA-main / CCA-aux use this because their
      upstream ``lrp_transforms`` consume the *post*-NAFTransform mean
      support; emitting "pre" would produce wrong LRP outputs even though
      the channel widths match.

    The trailing block is consumed by the model-local LRP Gaussian leaf to
    recover the upstream LRP input layout (``cat(mean_support, y_hat)``),
    enabling byte-for-byte transfer of upstream LRP weights.
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
        emit_mean_support: Union[bool, Literal["pre", "post"]] = False,
    ) -> None:
        super().__init__()
        self.mean_cc = mean_cc
        self.scale_cc = scale_cc
        self.mean_support_transform = mean_support_transform or nn.Identity()
        self.scale_support_transform = scale_support_transform or nn.Identity()
        self.side_split = int(side_split)
        self.emit_mean_support: Literal[False, "pre", "post"]
        if emit_mean_support is True:
            self.emit_mean_support = "pre"
        elif emit_mean_support is False:
            self.emit_mean_support = False
        elif emit_mean_support in ("pre", "post"):
            self.emit_mean_support = emit_mean_support
        else:
            raise ValueError(
                f"emit_mean_support must be False, True, 'pre', or 'post'; "
                f"got {emit_mean_support!r}"
            )
        if self.emit_mean_support and self.side_split <= 0:
            raise ValueError(
                "emit_mean_support requires side_split > 0 to recover the "
                "legacy mean_support layout cat(latent_means, *prev_y_hat)."
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
        mean_support = self.mean_support_transform(mean_in)
        mean = self.mean_cc(mean_support)
        scale = self.scale_cc(self.scale_support_transform(scale_in))
        out = torch.cat([scale, mean], dim=1)
        if self.emit_mean_support == "pre":
            out = torch.cat([out, mean_in], dim=1)
        elif self.emit_mean_support == "post":
            out = torch.cat([out, mean_support], dim=1)
        return out
