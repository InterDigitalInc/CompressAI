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
    support_count_fn: Optional[Callable[[int], int]] = None,
    side_in_context: bool = False,
    side_channels: int = 0,
) -> ChannelGroupsLatentCodec:
    """Assemble a :class:`ChannelGroupsLatentCodec` with per-slice modules.

    Generates the ``{"y0".."yK-1"}`` ``latent_codec`` dict and the
    ``{"y1".."yK-1"}`` ``channel_context`` dict (slice 0 has no channel
    context — it consumes ``side_params`` only). When
    ``side_in_context=True`` the ``channel_context`` dict additionally
    includes a ``"y0"`` entry whose input is just ``side_params``; the
    leaf for slice 0 then receives the head's output (already shaped
    ``2 * groups[0]``) instead of raw ``side_params``.

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
        channel-context module for slice ``k``. ``support_ch_k`` is the
        TOTAL channel count of the head's input — i.e., what
        :class:`ChannelGroupsLatentCodec._get_ctx_params` will hand it.
        For ELIC default mode (``side_in_context=False``) ``support_ch_k =
        sum(groups[:clamped_k])`` and only ``k >= 1`` entries are built.
        For Family 1 mode (``side_in_context=True``) ``support_ch_k =
        side_channels + sum(groups[:clamped_k])`` and a ``y0`` entry with
        ``support_ch_0 = side_channels`` is built too.
    max_support_slices
        Forwarded to :class:`ChannelGroupsLatentCodec`. Default ``-1`` uses
        all previous slices (ELIC / CCA-main behaviour).
    support_filter
        Forwarded to :class:`ChannelGroupsLatentCodec`. Used by CCA-aux for
        skip-most-recent support selection.
    support_count_fn
        ``(k) -> int``. Override for the number of prior slices that
        ``channel_context.y{k}`` will see at *runtime*. Required when
        ``support_filter`` selects a non-default count (e.g., CCA-aux's
        skip-most-recent ``lambda k: max(k - 1, 0)``); the factory uses
        this count when sizing the channel_context heads. Defaults to
        ``min(k, max_support_slices)`` (or ``k`` when
        ``max_support_slices < 0``), matching ``ChannelGroupsLatentCodec``'s
        own clamp logic when ``support_filter`` is unset.
    side_in_context
        Forwarded to :class:`ChannelGroupsLatentCodec`. When ``True`` the
        ``channel_context`` for ``y0`` consumes ``side_params`` and
        downstream ``y_k`` heads receive ``cat(side_params, prev_y_hat)``.
    side_channels
        Width of ``side_params`` (= hyper-synthesis output channel count).
        Required when ``side_in_context=True`` so the factory can size
        ``support_ch`` correctly.
    """
    if channel_context_factory is None:
        channel_context_factory = lambda *_: nn.Identity()  # noqa: E731
    if side_in_context and side_channels <= 0:
        raise ValueError(
            "side_in_context=True requires side_channels > 0 so the factory "
            "can size the channel_context heads (== side_channels for k=0; "
            "side_channels + sum(groups[:k]) clamped, for k>=1)."
        )

    K = len(groups)

    def _default_support_count(k: int) -> int:
        if max_support_slices < 0:
            return k
        return min(k, max_support_slices)

    _support_count = support_count_fn or _default_support_count

    def _support_ch(k: int) -> int:
        prior_ch = sum(groups[: _support_count(k)])
        if side_in_context:
            return side_channels + prior_ch
        return prior_ch

    if side_in_context:
        # y0 entry: head sees only side_params (no prev_y_hat yet).
        ctx_keys = range(0, K)
    else:
        # ELIC default: slice 0 bypasses channel_context entirely.
        ctx_keys = range(1, K)
    channel_context = {
        f"y{k}": channel_context_factory(k, groups[k], _support_ch(k)) for k in ctx_keys
    }
    latent_codec = {f"y{k}": leaf_factory(k, groups[k]) for k in range(K)}

    return ChannelGroupsLatentCodec(
        latent_codec=latent_codec,
        channel_context=channel_context,
        groups=list(groups),
        max_support_slices=max_support_slices,
        support_filter=support_filter,
        side_in_context=side_in_context,
    )
