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

"""Channel-slice support helpers shared by Family 1 codecs.

These functions support the Family 1 (pure 1-pass channel-slice) entropy
models — STF / WACNN / TCM / CCA / DCAE / MambaVC. They sit alongside the
latent-codec primitives that consume them.
``_DEFAULT_NUM_SLICES_PREFIX`` reflects the containerised state-dict layout
used by :class:`~compressai.latent_codecs.ChannelGroupsLatentCodec`.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch.nn as nn

from torch import Tensor

from compressai.models.utils import conv

__all__ = [
    "infer_max_support_slices",
    "infer_num_slices",
    "lrp_support_channels",
    "make_entropy_transform",
    "slice_support_channels",
]


# Post-refactor state-dict layout: ``HyperpriorLatentCodec`` exposes
# ``ChannelGroupsLatentCodec`` as ``self.y`` (the inner ``self.latent_codec``
# dict is not a registered nn.Module), so the channel-context entries live
# under ``latent_codec.y.channel_context.y{k}``. Slice 0 has no channel
# context entry by default (``side_in_context=False`` ELIC mode); Family 1
# ``side_in_context=True`` mode adds a ``y0`` entry whose presence triggers
# the auto-detection in :func:`infer_num_slices`.
_DEFAULT_NUM_SLICES_PREFIX = "latent_codec.y.channel_context.y"
_DEFAULT_KEY_SUFFIX = ".mean_cc.0.weight"


def slice_support_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * index
    return latent_channels + slice_channels * min(index, max_support_slices)


def lrp_support_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * (index + 1)
    return latent_channels + slice_channels * min(index + 1, max_support_slices + 1)


def make_entropy_transform(
    in_channels: int,
    out_channels: int,
    *,
    widths: Sequence[int] = (224, 128),
) -> nn.Sequential:
    """Stack of stride-1 3x3 convs with GELU activations.

    Used as the ``mean_cc`` / ``scale_cc`` per-slice heads (and as ``lrp_transform``)
    by every Family 1 channel-slice model. ``widths`` specifies hidden conv
    widths and defaults to the TCM / CCA / Mamba 3-conv stack
    ``(224, 128)``; pass ``widths=(224, 176, 128, 64)`` for the STF / WACNN
    5-conv stack.
    """
    layers: list[nn.Module] = []
    prev = in_channels
    for width in widths:
        layers.append(conv(prev, width, stride=1, kernel_size=3))
        layers.append(nn.GELU())
        prev = width
    layers.append(conv(prev, out_channels, stride=1, kernel_size=3))
    return nn.Sequential(*layers)


def infer_num_slices(
    state_dict: Dict[str, Tensor],
    *,
    prefix: str = _DEFAULT_NUM_SLICES_PREFIX,
    suffix: str = _DEFAULT_KEY_SUFFIX,
) -> int:
    """Count distinct ``y{k}`` channel-context entries in ``state_dict``.

    Two layouts are supported:

    - ELIC default: channel_context starts at ``y1`` (slice 0 bypasses it),
      so the count returned is ``num_slices - 1`` and we add ``1`` to recover
      ``num_slices``.
    - Family 1 ``side_in_context=True``: channel_context covers every
      slice including ``y0``, so the count is already ``num_slices``.

    The two cases are auto-detected by whether ``y0`` appears in the matched
    keys.
    """
    slice_indices = {
        int(key[len(prefix) :].split(".", 1)[0])
        for key in state_dict
        if key.startswith(prefix) and key.endswith(suffix)
    }
    if not slice_indices:
        return 0
    if 0 in slice_indices:
        return len(slice_indices)
    return len(slice_indices) + 1


def infer_max_support_slices(
    state_dict: Dict[str, Tensor],
    latent_channels: int,
    num_slices: int,
    *,
    prefix: str = _DEFAULT_NUM_SLICES_PREFIX,
    suffix: str = _DEFAULT_KEY_SUFFIX,
    extra_factor: int = 1,
) -> int:
    """Infer ``max_support_slices`` from the input width of the ``mean_cc``
    first conv. ``extra_factor`` accounts for application-layer heads (e.g.,
    DCAE / SAAF) that prepend additional copies of the latent
    (``M*extra + slice_channels*N``); default ``1`` covers Family 1 models
    whose ``mean_cc`` only sees the previous-slice support.
    """
    slice_channels = latent_channels // num_slices
    matching = [
        tensor.size(1)
        for key, tensor in state_dict.items()
        if key.startswith(prefix) and key.endswith(suffix)
    ]
    if not matching:
        return 0
    max_input_channels = max(matching)
    return max(
        0, (max_input_channels - extra_factor * latent_channels) // slice_channels
    )
