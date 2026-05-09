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

from typing import Callable, List, Optional

import torch.nn as nn

from torch import Tensor

from compressai.latent_codecs import ChannelGroupsLatentCodec
from compressai.latent_codecs.base import LatentCodec

__all__ = [
    "build_channel_slice_codec",
]


def build_channel_slice_codec(
    *,
    groups: List[int],
    leaf_factory: Callable[[int, int], LatentCodec],
    channel_context_factory: Optional[Callable[[int, int, int], nn.Module]] = None,
    max_support_slices: int = -1,
    support_filter: Optional[Callable[[int, List[Tensor]], List[Tensor]]] = None,
) -> ChannelGroupsLatentCodec:
    """Assemble a :class:`ChannelGroupsLatentCodec` with per-slice modules.

    Generates the ``{"y0".."yK-1"}`` ``latent_codec`` dict and the
    ``{"y1".."yK-1"}`` ``channel_context`` dict (slice 0 has no channel
    context — it consumes ``side_params`` only).

    Parameters
    ----------
    groups
        Per-slice channel counts. Use ``[M // K] * K`` for equal slices
        (STF / WACNN / TCM) or a custom list for variable-size slices (CCA).
    leaf_factory
        ``(k, slice_ch_k) -> LatentCodec``. Constructs the leaf for slice
        ``k`` — typically :class:`LRPGaussianLatentCodec` or
        :class:`GaussianConditionalLatentCodec`.
    channel_context_factory
        ``(k, slice_ch_k, support_ch_k) -> nn.Module``. Constructs the
        channel-context module for slice ``k`` (``k >= 1``). ``support_ch_k``
        is the total channel count of the previous slices that will be fed
        in (post ``max_support_slices`` clamp). Default ``None`` uses
        :class:`~torch.nn.Identity`, which is rarely useful in practice but
        keeps the API parallel with ``leaf_factory``.
    max_support_slices
        Forwarded to :class:`ChannelGroupsLatentCodec`. Default ``-1`` uses
        all previous slices (ELIC / CCA-main behaviour).
    support_filter
        Forwarded to :class:`ChannelGroupsLatentCodec`. Used by CCA-aux for
        skip-most-recent support selection.
    """
    if channel_context_factory is None:
        channel_context_factory = lambda *_: nn.Identity()  # noqa: E731

    K = len(groups)

    def _support_ch(k: int) -> int:
        if max_support_slices < 0:
            count = k
        else:
            count = min(k, max_support_slices)
        return sum(groups[:count])

    channel_context = {
        f"y{k}": channel_context_factory(k, groups[k], _support_ch(k))
        for k in range(1, K)
    }
    latent_codec = {f"y{k}": leaf_factory(k, groups[k]) for k in range(K)}

    return ChannelGroupsLatentCodec(
        latent_codec=latent_codec,
        channel_context=channel_context,
        groups=list(groups),
        max_support_slices=max_support_slices,
        support_filter=support_filter,
    )
