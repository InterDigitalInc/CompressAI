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

    Forward output is ``cat([mean_cc(...), scale_cc(...)], dim=1)`` of shape
    ``(B, 2 * slice_ch, H, W)``. Optional ``mean_support_transform`` /
    ``scale_support_transform`` run independently on the input before the
    sub-networks (used for SWAtten in TCM and NAFTransform in CCA).
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
    ) -> None:
        super().__init__()
        self.mean_cc = mean_cc
        self.scale_cc = scale_cc
        self.mean_support_transform = mean_support_transform or nn.Identity()
        self.scale_support_transform = scale_support_transform or nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        mean = self.mean_cc(self.mean_support_transform(x))
        scale = self.scale_cc(self.scale_support_transform(x))
        return torch.cat([mean, scale], dim=1)


def build_mean_scale_head(
    slice_ch: int,
    support_ch: int,
    *,
    widths: Sequence[int] = (224, 128),
    support_transform_factory: Optional[Callable[[int, int], nn.Module]] = None,
) -> MeanScaleContextHead:
    """Construct a :class:`MeanScaleContextHead` with default conv-stack heads.

    Parameters
    ----------
    slice_ch
        Channel count of the slice being predicted (per-sub-head output).
    support_ch
        Input channel count to ``mean_cc`` / ``scale_cc`` (post-support-
        transform). Caller is responsible for accounting for any extra
        channels that the application's wiring concatenates upstream.
    widths
        Hidden conv widths inside the ``mean_cc`` / ``scale_cc`` Sequentials.
        STF / WACNN use ``(224, 176, 128, 64)``; TCM / CCA use
        ``(224, 128)``.
    support_transform_factory
        ``(in_ch, out_ch) -> nn.Module``. When supplied, builds independent
        instances for the mean and scale paths (e.g., per-slice SWAtten in
        TCM or NAFTransform in CCA). Both transforms are expected to
        preserve channel count.
    """
    mean_cc = make_entropy_transform(support_ch, slice_ch, widths=widths)
    scale_cc = make_entropy_transform(support_ch, slice_ch, widths=widths)
    mean_support: Optional[nn.Module]
    scale_support: Optional[nn.Module]
    if support_transform_factory is not None:
        mean_support = support_transform_factory(support_ch, support_ch)
        scale_support = support_transform_factory(support_ch, support_ch)
    else:
        mean_support = None
        scale_support = None
    return MeanScaleContextHead(
        mean_cc=mean_cc,
        scale_cc=scale_cc,
        mean_support_transform=mean_support,
        scale_support_transform=scale_support,
    )
