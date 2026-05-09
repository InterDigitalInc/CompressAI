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

from ._hyper_synthesis import DualHyperSynthesis
from .base import LatentCodec
from .channel_groups import ChannelGroupsLatentCodec
from .channel_slice import ChannelSliceLatentCodec
from .checkerboard import CheckerboardLatentCodec
from .entropy_bottleneck import EntropyBottleneckLatentCodec
from .gain import GainHyperLatentCodec, GainHyperpriorLatentCodec
from .gaussian_conditional import GaussianConditionalLatentCodec, LRPGaussianLatentCodec
from .hyper import HyperLatentCodec
from .hyperprior import HyperpriorLatentCodec
from .rasterscan import RasterScanLatentCodec

__all__ = [
    "LatentCodec",
    "ChannelGroupsLatentCodec",
    "ChannelSliceLatentCodec",
    "CheckerboardLatentCodec",
    "DualHyperSynthesis",
    "EntropyBottleneckLatentCodec",
    "GainHyperLatentCodec",
    "GainHyperpriorLatentCodec",
    "GaussianConditionalLatentCodec",
    "HyperLatentCodec",
    "HyperpriorLatentCodec",
    "LRPGaussianLatentCodec",
    "RasterScanLatentCodec",
]


# ----------------------------------------------------------------------------
# Family 1 wiring (STF / WACNN / TCM / CCA / DCAE / MambaVC)
# ----------------------------------------------------------------------------
#
# "Family 1" is the set of channel-slice models that share the same outer
# entropy-stack shape:
#
#     HyperpriorLatentCodec(
#         h_a=h_a,
#         h_s=DualHyperSynthesis(h_mean_s, h_scale_s),  # cat(mean_s, scale_s)
#         latent_codec={
#             "z": EntropyBottleneckLatentCodec(EntropyBottleneck(N), ...),
#             "y": ChannelGroupsLatentCodec(  # side_in_context=True mode
#                 latent_codec={"y0": LRPGaussianLatentCodec(...), ...},
#                 channel_context={"y0": MeanScaleContextHead(...), ...},
#                 groups=[M//K]*K,
#                 max_support_slices=MS,
#                 side_in_context=True,
#             ),
#         },
#     )
#
# Compared to the ELIC-style channel-slice wiring it differs in three
# places, all reproducible through optional kwargs on the upstream codecs:
#
# 1. Two parallel ``h_s`` heads instead of one — DualHyperSynthesis cats
#    them into a single ``side_params`` tensor of width ``2*M``.
# 2. ``ChannelGroupsLatentCodec(side_in_context=True)`` routes
#    ``side_params`` into every channel_context head (including ``y0``)
#    instead of only handing it to the leaves; the head is then
#    responsible for re-splitting ``side_params`` into mean / scale.
# 3. The leaf is :class:`LRPGaussianLatentCodec` (mostly), which adds a
#    learned residual prediction on top of ``y_hat``. With matching
#    ``mean_support_trail_channels`` the leaf reads the LRP input from a
#    trailing block of ``ctx_params`` produced by the head's
#    ``emit_mean_support=True`` mode, recovering the upstream
#    ``cat(latent_means, *prev_y_hat, y_hat)`` layout for byte-for-byte
#    weight transfer.
#
# Application-layer helpers in
# :mod:`compressai.models._helpers.channel_slice` and
# :mod:`compressai.models._helpers.channel_context`
# (``build_channel_slice_codec``, ``MeanScaleContextHead``,
# ``build_mean_scale_head``) wire these pieces declaratively. Per-model
# variations stay in the kwargs:
#
# - **STF / WACNN**: 5-conv cc heads ``widths=(224, 176, 128, 64)``, no
#   support transform.
# - **TCM**: 3-conv cc heads ``widths=(224, 128)``,
#   ``support_transform_factory=SWAtten`` (independent windowed-attention
#   transforms per mean / scale path).
# - **CCA-main**: variable-length slices (``groups=resolved_slice_sizes``),
#   ``support_transform_factory=NAFTransform``,
#   ``EntropyBottleneckLatentCodec(quantizer="ste")`` for the ``z`` leaf.
# - **CCA-aux**: lives outside the hyperprior container (separate
#   ``ChannelGroupsLatentCodec``), uses ``support_filter`` for
#   skip-most-recent prior selection, and mixes
#   :class:`LRPGaussianLatentCodec` (early slices) with
#   :class:`GaussianConditionalLatentCodec` (last two slices).
# - **DCAE / MambaVC**: future Family 1 follow-ups; same shape, different
#   support transforms.
#
# See :mod:`compressai.models.stf` and :mod:`compressai.models.tcm` for
# end-to-end examples.
