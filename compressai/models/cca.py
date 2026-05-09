# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/CVL-UESTC/CCA
# (originally distributed under the MIT License). The upstream copyright
# notice is preserved in that repository; modifications by InterDigital
# Communications, Inc. are released under the BSD 3-Clause Clear License
# terms below.

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

"""Causal Context Adjustment (CCA) standalone autoencoder.

Mirror of the upstream ``LICAutoencoder`` from
M. Han, S. Jiang, S. Li, X. Deng, M. Xu, C. Zhu, S. Gu:
`"Causal Context Adjustment Loss for Learned Image Compression"
<https://arxiv.org/abs/2410.04847>`_, NeurIPS 2024.

Family 1 wiring (see :mod:`compressai.latent_codecs.__init__`): the main
entropy stack is a fully containerised :class:`HyperpriorLatentCodec`
running ``side_in_context=True`` with variable-length slice groups and
per-slice :class:`_NAFTransform` support transforms. The optional
auxiliary CCA branch (:class:`_CCAAuxEntropyModel`) is a separate
``nn.Module`` that re-encodes ``y`` with the skip-most-recent
``support_filter`` selection used by
:class:`compressai.losses.CCARateDistortionLoss` to align the causal
context with the rate-distortion objective.
"""

from __future__ import annotations

import math

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    DualHyperSynthesis,
    EntropyBottleneckLatentCodec,
    GaussianConditionalLatentCodec,
    HyperpriorLatentCodec,
    LRPGaussianLatentCodec,
)
from compressai.latent_codecs._slice_helpers import make_entropy_transform
from compressai.layers.layers import conv1x1
from compressai.models._helpers.channel_context import build_mean_scale_head
from compressai.models._helpers.channel_slice import build_channel_slice_codec
from compressai.models.base import CompressionModel, get_scale_table
from compressai.models.sensetime import ResidualBottleneckBlock
from compressai.models.utils import conv, deconv
from compressai.registry import register_model

__all__ = [
    "CCAModel",
    "convert_upstream_cca_state_dict",
]


# ----------------------------------------------------------------------------
# Slice-size resolver.
# ----------------------------------------------------------------------------


def _resolve_slice_sizes(
    latent_channels: int, slice_proportions: Sequence[int]
) -> List[int]:
    if len(slice_proportions) == 0:
        raise ValueError("slice_proportions must contain at least one entry")
    total = sum(slice_proportions)
    if total <= 0:
        raise ValueError("slice_proportions must sum to a positive integer")
    sizes = [
        int(math.floor(latent_channels * proportion / total))
        for proportion in slice_proportions
    ]
    sizes[-1] += latent_channels - sum(sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("resolved slice sizes must all be positive")
    return sizes


# ----------------------------------------------------------------------------
# NAF (Non-linear Activation Free) building blocks
# ----------------------------------------------------------------------------


class _SimpleGate(nn.Module):
    def forward(self, input_tensor: Tensor) -> Tensor:
        gate_tensor, value_tensor = input_tensor.chunk(2, dim=1)
        return gate_tensor * value_tensor


class _NAFBlock(nn.Module):
    """Non-linear Activation Free residual block.

    Used by both the CCA entropy-model auxiliary transforms and the CCA
    image-compression model's analysis / synthesis stacks. State-dict keys
    (``norm1`` / ``pointwise_depthwise`` / ``channel_attention`` /
    ``project`` / ``feed_forward`` / ``beta`` / ``gamma``) match upstream
    after :func:`convert_upstream_cca_state_dict` so released checkpoints
    load 1:1.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        from timm.layers import LayerNorm2d

        expanded_channels = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.pointwise_depthwise = nn.Sequential(
            conv1x1(channels, expanded_channels),
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                padding=1,
                groups=expanded_channels,
            ),
        )
        self.gate = _SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(channels, channels),
        )
        self.project = conv1x1(channels, channels)
        self.norm2 = LayerNorm2d(channels)
        self.feed_forward = nn.Sequential(
            conv1x1(channels, expanded_channels),
            _SimpleGate(),
            conv1x1(channels, channels),
        )
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.norm1(input_tensor)
        output = self.pointwise_depthwise(output)
        output = self.gate(output)
        output = output * self.channel_attention(output)
        output = self.project(output)
        output = input_tensor + self.beta * output
        return output + self.gamma * self.feed_forward(self.norm2(output))


class _NAFTransform(nn.Module):
    """``Conv1x1 -> NAFBlock x N -> Conv1x1`` per-slice support transform."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.input_projection = conv1x1(input_channels, hidden_channels)
        self.blocks = nn.Sequential(
            *(_NAFBlock(hidden_channels) for _ in range(num_layers))
        )
        self.output_projection = conv1x1(hidden_channels, output_channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.input_projection(input_tensor)
        return self.output_projection(output + self.blocks(output))


# ----------------------------------------------------------------------------
# Analysis / synthesis transforms (NAFBlock + ResidualBottleneckBlock).
# ----------------------------------------------------------------------------


class _CCAEncoder(nn.Module):
    """NAFBlock + ResidualBottleneckBlock analysis transform (4 strides)."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        stage_dims: Sequence[int],
        stage_layers: Sequence[int],
    ) -> None:
        super().__init__()
        if len(stage_dims) != len(stage_layers):
            raise ValueError("stage_dims and stage_layers must have matching length")
        self.depth = len(stage_dims)
        all_dims = [in_channels, *stage_dims, latent_channels]
        self.down = nn.ModuleList(
            conv(all_dims[index], all_dims[index + 1], kernel_size=5, stride=2)
            for index in range(self.depth + 1)
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                *(
                    ResidualBottleneckBlock(stage_dims[index], stage_dims[index])
                    for _ in range(3)
                ),
                *(_NAFBlock(stage_dims[index]) for _ in range(stage_layers[index])),
            )
            for index in range(self.depth)
        )

    def forward(self, x: Tensor) -> Tensor:
        for index in range(self.depth):
            x = self.down[index](x)
            x = self.blocks[index](x)
        return self.down[self.depth](x)


class _CCADecoder(nn.Module):
    """NAFBlock + ResidualBottleneckBlock synthesis transform (4 strides)."""

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        stage_dims: Sequence[int],
        stage_layers: Sequence[int],
    ) -> None:
        super().__init__()
        if len(stage_dims) != len(stage_layers):
            raise ValueError("stage_dims and stage_layers must have matching length")
        self.depth = len(stage_dims)
        all_dims = [out_channels, *stage_dims, latent_channels]
        self.up = nn.ModuleList(
            deconv(all_dims[index + 1], all_dims[index], kernel_size=5, stride=2)
            for index in reversed(range(self.depth + 1))
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                *(_NAFBlock(stage_dims[index]) for _ in range(stage_layers[index])),
                *(
                    ResidualBottleneckBlock(stage_dims[index], stage_dims[index])
                    for _ in range(3)
                ),
            )
            for index in reversed(range(self.depth))
        )

    def forward(self, x: Tensor) -> Tensor:
        for index in range(self.depth):
            x = self.up[index](x)
            x = self.blocks[index](x)
        return self.up[self.depth](x)


# ----------------------------------------------------------------------------
# Family 1 entropy-stack builders (main + auxiliary).
# ----------------------------------------------------------------------------


def _build_cca_main_latent_codec(
    *,
    M: int,
    hyper_channels: int,
    slice_sizes: Sequence[int],
    em_hidden_channels: int,
    em_num_layers: int,
    h_a: nn.Module,
    h_mean_s: nn.Module,
    h_scale_s: nn.Module,
) -> HyperpriorLatentCodec:
    """Main entropy stack: ``HyperpriorLatentCodec`` wrapping
    ``DualHyperSynthesis`` and a per-slice ``ChannelGroupsLatentCodec``.

    Distinctive choices vs. STF/TCM (other Family 1 models):

    - ``groups`` is a variable-length list (resolved from
      ``slice_proportions``); STF / WACNN / TCM use uniform ``[M//K]*K``.
    - ``support_transform_factory`` builds a per-slice
      :class:`_NAFTransform` for both mean and scale paths (vs. STF
      identity / TCM SWAtten).
    - The leaf is :class:`LRPGaussianLatentCodec` with
      ``mean_support_trail_channels`` matching
      ``M + sum(slice_sizes[:k])``, paired with
      ``MeanScaleContextHead(emit_mean_support="post")`` so the LRP head
      receives the *post*-NAFTransform mean support — replicating the
      upstream LIC LRP layout for byte-for-byte weight transfer.
    - The ``z`` leaf uses ``EntropyBottleneckLatentCodec(quantizer="ste")``
      to recover upstream's ``quantize_ste(z - z_offset) + z_offset``
      behaviour without a model-side hack.
    """
    cumulative = list(_cumsum_with_zero(slice_sizes))
    side_channels = 2 * M
    K = len(slice_sizes)

    def _support_count(k: int) -> int:
        # use-all-prior; matches upstream LIC main path (no skip).
        return k

    def _mean_support_ch(k: int) -> int:
        # cat(latent_means(M), *prev_y_hat(sum(slice_sizes[:k]))).
        return M + cumulative[k]

    def _leaf(k: int, slice_ch: int) -> LRPGaussianLatentCodec:
        ms_ch = _mean_support_ch(k)
        return LRPGaussianLatentCodec(
            lrp_transform=_make_cca_head(
                ms_ch + slice_ch,  # cat(mean_support, y_hat)
                em_hidden_channels,
                slice_ch,
            ),
            mean_support_trail_channels=ms_ch,
            quantizer="ste",
        )

    def _naf_factory(c_in: int, c_out: int) -> nn.Module:
        return _NAFTransform(c_in, c_out, em_hidden_channels, em_num_layers)

    def _channel_context(_k: int, slice_ch: int, support_ch: int) -> nn.Module:
        return build_mean_scale_head(
            slice_ch=slice_ch,
            support_ch=support_ch,
            widths=(em_hidden_channels, 128),
            side_split=M,
            emit_mean_support="post",
            support_transform_factory=_naf_factory,
        )

    if K == 0:
        raise ValueError("slice_sizes must contain at least one entry")

    return HyperpriorLatentCodec(
        h_a=h_a,
        h_s=DualHyperSynthesis(h_mean_s, h_scale_s),
        latent_codec={
            "z": EntropyBottleneckLatentCodec(
                entropy_bottleneck=EntropyBottleneck(hyper_channels),
                quantizer="ste",
            ),
            "y": build_channel_slice_codec(
                groups=list(slice_sizes),
                side_channels=side_channels,
                side_in_context=True,
                max_support_slices=-1,
                support_count_fn=_support_count,
                leaf_factory=_leaf,
                channel_context_factory=_channel_context,
            ),
        },
    )


def _make_cca_head(
    in_channels: int, hidden_channels: int, out_channels: int
) -> nn.Sequential:
    """Three-conv stack ``in -> hidden -> 128 -> out`` (kernel 3, stride 1).

    Matches upstream ``mean_cc_transforms[k]`` / ``lrp_transforms[k]``
    layout. Wraps :func:`make_entropy_transform` with the CCA-specific
    ``widths=(hidden_channels, 128)``.
    """
    return make_entropy_transform(
        in_channels, out_channels, widths=(hidden_channels, 128)
    )


def _cumsum_with_zero(values: Sequence[int]) -> List[int]:
    """Return ``[0, values[0], values[0]+values[1], ...]`` (length ``len+1``)."""
    out = [0]
    running = 0
    for value in values:
        running += int(value)
        out.append(running)
    return out


class _CCAAuxEntropyModel(nn.Module):
    """Auxiliary CCA entropy branch (skip-most-recent-slice support).

    Produces the ``y_aux`` (factorised) and ``y_cca`` (Gaussian-conditional)
    likelihoods used by :class:`compressai.losses.CCARateDistortionLoss`.

    Mirrors the upstream ``AuxEntropyModel`` in
    ``candidate/CCA/models/aux_em.py``: for slice ``i`` the support is
    ``cat(latent_means, *y_hat_slices[: max(i - 1, 0)])`` (i.e., skip the
    *most recent* decoded slice). This is wired declaratively through
    :func:`build_channel_slice_codec` with
    ``support_filter=lambda k, prior: prior[: max(k - 1, 0)]`` and a
    matching ``support_count_fn`` to size the channel-context heads.

    Although upstream only *uses* the LRP path on the first ``num_slices -
    2`` slices, the published checkpoints carry LRP weights for *all*
    slices. To strict-load those checkpoints every leaf is a
    :class:`LRPGaussianLatentCodec`; the LRP applied to the trailing two
    slices is benign (those slices' ``y_hat`` is excluded from every
    later slice's support_filter, so it never feeds back into the
    likelihoods).
    """

    def __init__(
        self,
        latent_channels: int,
        slice_sizes: Sequence[int],
        hidden_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.slice_sizes: List[int] = list(map(int, slice_sizes))
        self.num_slices = len(self.slice_sizes)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)

        cumulative = _cumsum_with_zero(self.slice_sizes)
        side_channels = 2 * self.latent_channels

        def _support_count(k: int) -> int:
            return max(k - 1, 0)

        def _support_filter(k: int, prior: List[Tensor]) -> List[Tensor]:
            return prior[: max(k - 1, 0)]

        def _mean_support_ch(k: int) -> int:
            return self.latent_channels + cumulative[_support_count(k)]

        def _leaf(k: int, slice_ch: int) -> LRPGaussianLatentCodec:
            ms_ch = _mean_support_ch(k)
            return LRPGaussianLatentCodec(
                lrp_transform=_make_cca_head(
                    ms_ch + slice_ch,
                    self.hidden_channels,
                    slice_ch,
                ),
                mean_support_trail_channels=ms_ch,
                quantizer="ste",
            )

        def _naf_factory(c_in: int, c_out: int) -> nn.Module:
            return _NAFTransform(c_in, c_out, self.hidden_channels, self.num_layers)

        def _channel_context(_k: int, slice_ch: int, support_ch: int) -> nn.Module:
            return build_mean_scale_head(
                slice_ch=slice_ch,
                support_ch=support_ch,
                widths=(self.hidden_channels, 128),
                side_split=self.latent_channels,
                emit_mean_support="post",
                support_transform_factory=_naf_factory,
            )

        self.y_entropy_bottleneck = EntropyBottleneck(self.latent_channels)
        self.inner_codec = build_channel_slice_codec(
            groups=list(self.slice_sizes),
            side_channels=side_channels,
            side_in_context=True,
            max_support_slices=-1,
            support_filter=_support_filter,
            support_count_fn=_support_count,
            leaf_factory=_leaf,
            channel_context_factory=_channel_context,
        )

    def forward(
        self,
        y: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Dict[str, Tensor]:
        _, y_aux_likelihoods = self.y_entropy_bottleneck(y)
        side_params = torch.cat([latent_means, latent_scales], dim=1)
        inner_out = self.inner_codec(y, side_params)
        return {
            "y_aux": y_aux_likelihoods,
            "y_cca": inner_out["likelihoods"]["y"],
        }


# ----------------------------------------------------------------------------
# Top-level CCAModel.
# ----------------------------------------------------------------------------


@register_model("cca")
class CCAModel(CompressionModel):
    r"""Causal Context Adjustment standalone autoencoder.

    Mirrors the upstream ``LICAutoencoder`` from M. Han et al., NeurIPS 2024
    (`Causal Context Adjustment Loss for Learned Image Compression
    <https://arxiv.org/abs/2410.04847>`_).

    The entropy stack is a Family 1 :class:`HyperpriorLatentCodec` (see
    :mod:`compressai.latent_codecs.__init__` for the full pattern) with
    variable-length channel slices (``slice_proportions``), per-slice
    :class:`_NAFTransform` support transforms, and a STE-quantised ``z``
    leaf. When ``cca_training=True`` an auxiliary
    :class:`_CCAAuxEntropyModel` branch is added that produces ``y_aux`` /
    ``y_cca`` likelihoods consumed by
    :class:`compressai.losses.CCARateDistortionLoss`.

    Args:
        latent_channels: Number of channels in the latent (``M``).
        hyper_channels: Number of channels in the hyper-latent (``N_z``).
        slice_proportions: Per-slice channel proportions; the actual slice
            channel widths are computed as
            ``floor(latent_channels * p / sum(p))`` with the residual added
            to the last slice. Pass ``[1] * num_slices`` for equal-sized
            slices; pass ``[8, 28, 56, 92, 136]`` to reproduce the upstream
            published M=320 layout.
        encoder_dims: Per-stage feature widths for the analysis transform
            (3 stages by default).
        encoder_layers: Per-stage NAFBlock counts for the analysis transform.
        em_hidden_channels: Hidden width inside the per-slice NAFTransforms
            and channel-context heads.
        em_num_layers: NAFBlock count inside each per-slice NAFTransform.
        cca_training: When ``True``, allocate the auxiliary CCA entropy
            branch so that ``forward`` populates ``aux_likelihoods``.
    """

    def __init__(
        self,
        latent_channels: int = 320,
        hyper_channels: int = 192,
        slice_proportions: Sequence[int] = (8, 28, 56, 92, 136),
        encoder_dims: Sequence[int] = (192, 224, 256),
        encoder_layers: Sequence[int] = (4, 4, 4),
        em_hidden_channels: int = 224,
        em_num_layers: int = 4,
        cca_training: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        encoder_dims = tuple(encoder_dims)
        encoder_layers = tuple(encoder_layers)
        slice_proportions = tuple(int(value) for value in slice_proportions)

        self.M = int(latent_channels)
        self.N = int(hyper_channels)
        self.encoder_dims = encoder_dims
        self.encoder_layers = encoder_layers
        self.slice_proportions = slice_proportions
        self.em_hidden_channels = int(em_hidden_channels)
        self.em_num_layers = int(em_num_layers)
        self.cca_training = bool(cca_training)

        self.slice_sizes: List[int] = _resolve_slice_sizes(self.M, slice_proportions)
        self.num_slices = len(self.slice_sizes)

        self.g_a = _CCAEncoder(3, self.M, encoder_dims, encoder_layers)
        self.g_s = _CCADecoder(3, self.M, encoder_dims, encoder_layers)

        last_encoder_dim = encoder_dims[-1]
        h_a = nn.Sequential(
            conv(self.M, last_encoder_dim, kernel_size=3, stride=1),
            nn.GELU(),
            conv(last_encoder_dim, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            conv(last_encoder_dim, self.N, kernel_size=5, stride=2),
        )
        h_mean_s = nn.Sequential(
            deconv(self.N, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            deconv(last_encoder_dim, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            deconv(last_encoder_dim, self.M, kernel_size=3, stride=1),
        )
        h_scale_s = nn.Sequential(
            deconv(self.N, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            deconv(last_encoder_dim, last_encoder_dim, kernel_size=5, stride=2),
            nn.GELU(),
            deconv(last_encoder_dim, self.M, kernel_size=3, stride=1),
        )

        self.latent_codec = _build_cca_main_latent_codec(
            M=self.M,
            hyper_channels=self.N,
            slice_sizes=self.slice_sizes,
            em_hidden_channels=self.em_hidden_channels,
            em_num_layers=self.em_num_layers,
            h_a=h_a,
            h_mean_s=h_mean_s,
            h_scale_s=h_scale_s,
        )

        if self.cca_training:
            self.aux_entropy_model = _CCAAuxEntropyModel(
                self.M,
                self.slice_sizes,
                self.em_hidden_channels,
                self.em_num_layers,
            )

    def forward(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        result: Dict[str, object] = {
            "y": y,
            "x_hat": self.g_s(y_out["y_hat"]),
            "likelihoods": y_out["likelihoods"],
        }
        if self.cca_training:
            # ``self.latent_codec.h_s`` is the ``DualHyperSynthesis``; its
            # output is ``cat(latent_means, latent_scales)`` of width 2*M.
            # Recover them from the inner ``z`` round-trip so the aux
            # branch sees the same hyperprior context as the main path.
            z_out = self.latent_codec.latent_codec["z"](self.latent_codec.h_a(y))
            side_params = self.latent_codec.h_s(z_out["y_hat"])
            latent_means, latent_scales = torch.split(side_params, self.M, dim=1)
            result["aux_likelihoods"] = self.aux_entropy_model(
                y, latent_means, latent_scales
            )
        else:
            result["aux_likelihoods"] = None
        return result

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        y_out = self.latent_codec.compress(y)
        return {"strings": y_out["strings"], "shape": y_out["shape"]}

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        y_out = self.latent_codec.decompress(strings, shape)
        return {"x_hat": self.g_s(y_out["y_hat"]).clamp_(0, 1)}

    def update(
        self, scale_table: Optional[Tensor] = None, force: bool = False, **kwargs
    ) -> bool:
        if scale_table is None:
            scale_table = get_scale_table()
        return super().update(scale_table=scale_table, force=force, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        if _is_upstream_cca_state_dict(state_dict):
            state_dict = convert_upstream_cca_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "CCAModel":
        if _is_upstream_cca_state_dict(state_dict):
            state_dict = convert_upstream_cca_state_dict(state_dict)
        cfg = _infer_config_from_state_dict(state_dict)
        net = cls(**cfg)
        net.load_state_dict(state_dict)
        return net


# ----------------------------------------------------------------------------
# Upstream → compressai state-dict conversion.
# ----------------------------------------------------------------------------


# NAFBlock interior renames (upstream -> compressai). These are scoped to
# detected NAFBlock prefixes so they don't accidentally rewrite ``conv1`` in
# unrelated modules (e.g. ResidualBottleneckBlock has its own ``conv1``).
_NAF_BLOCK_RENAMES = {
    "dwconv.": "pointwise_depthwise.",
    "sca.": "channel_attention.",
    "FFN.": "feed_forward.",
    "conv1.": "project.",
}
# NAFTransform interior renames.
_NAF_TRANSFORM_RENAMES = {
    "in_conv.": "input_projection.",
    "out_conv.": "output_projection.",
}
# Top-level rename map applied AFTER NAFBlock / NAFTransform interior renames
# and BEFORE per-slice rerooting. Used for hyperprior backbone and aux module.
_TOPLEVEL_RENAMES: Dict[str, str] = {
    "aux_entropymodel.": "aux_entropy_model.",
    "h_a.": "latent_codec.h_a.",
    "h_mean_s.": "latent_codec.h_s.h_mean_s.",
    "h_scale_s.": "latent_codec.h_s.h_scale_s.",
    "z_entropy_bottleneck.": "latent_codec.z.entropy_bottleneck.",
}
# Upstream uses ``mean_NAF_transforms`` / ``scale_NAF_transforms``; this PR
# stores them at ``{mean,scale}_support_transform`` inside the channel-context
# head (singular per slice). Aliasing here keeps the per-slice rerooting pass
# uniform across main and aux branches.
_NAMED_PART_RENAMES: Dict[str, str] = {
    "mean_NAF_transforms.": "mean_support_transforms.",
    "scale_NAF_transforms.": "scale_support_transforms.",
}


def _is_upstream_cca_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic detector for upstream ``LICAutoencoder`` checkpoints."""
    for key in state_dict:
        if (
            key.startswith("mean_NAF_transforms.")
            or key.startswith("scale_NAF_transforms.")
            or key.startswith("aux_entropymodel.")
            or key.startswith("z_entropy_bottleneck.")
            or key.startswith("mean_cc_transforms.")
            or key.startswith("scale_cc_transforms.")
            or key.startswith("lrp_transforms.")
        ):
            return True
    return False


def _find_naf_block_prefixes(state_dict: Dict[str, Tensor]) -> List[str]:
    """Locate every NAFBlock instance by matching the ``.beta`` /  ``.gamma``
    / ``.dwconv.0.weight`` / ``.FFN.0.weight`` 4-tuple at the same scope.
    """
    suffix = ".beta"
    out: List[str] = []
    for key in state_dict:
        if not key.endswith(suffix):
            continue
        base = key[: -len(suffix)]
        if (
            f"{base}.gamma" in state_dict
            and f"{base}.dwconv.0.weight" in state_dict
            and f"{base}.FFN.0.weight" in state_dict
        ):
            out.append(base)
    return out


def _find_naf_transform_prefixes(state_dict: Dict[str, Tensor]) -> List[str]:
    """Locate every NAFTransform instance by matching the ``.in_conv.weight``
    / ``.out_conv.weight`` / ``.blocks.0.beta`` triple at the same scope.
    """
    suffix = ".in_conv.weight"
    out: List[str] = []
    for key in state_dict:
        if not key.endswith(suffix):
            continue
        base = key[: -len(suffix)]
        if (
            f"{base}.out_conv.weight" in state_dict
            and f"{base}.blocks.0.beta" in state_dict
        ):
            out.append(base)
    return out


def _strip_prefix(key: str, prefix: str) -> Optional[str]:
    return key[len(prefix) :] if key.startswith(prefix) else None


def _rename_with_table(
    key: str,
    base_prefixes: Sequence[str],
    rename_map: Dict[str, str],
) -> str:
    for base in base_prefixes:
        head = base + "."
        rest = _strip_prefix(key, head)
        if rest is None:
            continue
        for old, new in rename_map.items():
            inner = _strip_prefix(rest, old)
            if inner is not None:
                return head + new + inner
        return key
    return key


def _reroot_per_slice_keys(
    cleaned: Dict[str, Tensor],
    converted: Dict[str, Tensor],
    *,
    legacy_prefix: str,
    container_prefix: str,
    sub_name: str,
    num_slices: int,
    consume: List[str],
) -> None:
    """Move ``legacy_prefix.{k}.<...>`` keys to
    ``container_prefix.y{k}.sub_name.<...>``.

    Keys that match are removed from ``cleaned`` (recorded in ``consume``
    for a later bulk drop) and inserted into ``converted`` under the new
    path.
    """
    for key in list(cleaned.keys()):
        rest = _strip_prefix(key, legacy_prefix + ".")
        if rest is None:
            continue
        idx_str, _, tail = rest.partition(".")
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        if idx >= num_slices:
            continue
        new_key = (
            f"{container_prefix}.y{idx}.{sub_name}.{tail}"
            if tail
            else f"{container_prefix}.y{idx}.{sub_name}"
        )
        converted[new_key] = cleaned[key]
        consume.append(key)


def _replicate_gaussian_conditional(
    cleaned: Dict[str, Tensor],
    converted: Dict[str, Tensor],
    *,
    legacy_prefix: str,
    new_prefix: str,
    num_slices: int,
    consume: List[str],
) -> None:
    """Copy a single shared ``gaussian_conditional.<...>`` buffer set under
    every per-slice leaf so the per-slice
    :class:`GaussianConditionalLatentCodec` copies all strict-load.
    """
    for key in list(cleaned.keys()):
        tail = _strip_prefix(key, legacy_prefix + ".")
        if tail is None:
            continue
        for k in range(num_slices):
            new_key = f"{new_prefix}.y{k}.gaussian_conditional.{tail}"
            converted[new_key] = cleaned[key]
        consume.append(key)


def convert_upstream_cca_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream CCA ``LICAutoencoder`` state dict to the
    compressai layout produced by :class:`CCAModel`.

    Conversion runs three logical passes:

    1. Interior renames: ``NAFBlock`` (``dwconv`` → ``pointwise_depthwise``,
       etc.) and ``NAFTransform`` (``in_conv`` → ``input_projection``,
       etc.). Detection is by structural fingerprint
       (:func:`_find_naf_block_prefixes`) so the renames apply uniformly to
       NAFBlocks anywhere in the state dict (``g_a`` / ``g_s`` / per-slice
       support transforms / aux module).
    2. Top-level renames: ``aux_entropymodel`` → ``aux_entropy_model``,
       hyperprior backbone (``h_a`` / ``h_mean_s`` / ``h_scale_s``) and
       ``z_entropy_bottleneck`` are moved under ``latent_codec.*``;
       ``mean_NAF_transforms`` / ``scale_NAF_transforms`` are aliased to
       the singular ``{mean,scale}_support_transforms`` form so the
       per-slice rerooting in pass 3 only handles one name.
    3. Per-slice rerooting: ``mean_cc_transforms.{k}`` /
       ``scale_cc_transforms.{k}`` move to
       ``latent_codec.y.channel_context.y{k}.{mean,scale}_cc.*``;
       ``mean_support_transforms.{k}`` / ``scale_support_transforms.{k}``
       move to
       ``latent_codec.y.channel_context.y{k}.{mean,scale}_support_transform.*``;
       ``lrp_transforms.{k}`` moves to
       ``latent_codec.y.latent_codec.y{k}.lrp_transform.*``; the single
       shared ``gaussian_conditional.*`` buffer set is replicated under
       every per-slice leaf
       (``latent_codec.y.latent_codec.y{k}.gaussian_conditional.*``). The
       same rerooting is applied to ``aux_entropy_model.*`` (after the
       top-level rename) under ``aux_entropy_model.inner_codec.*``.

    The returned dict can be loaded by :meth:`CCAModel.from_state_dict`,
    which auto-detects the upstream layout via
    :func:`_is_upstream_cca_state_dict`, so direct invocation is only
    needed when persisting the converted dict.
    """
    naf_blocks = _find_naf_block_prefixes(state_dict)
    naf_transforms = _find_naf_transform_prefixes(state_dict)

    # Pass 1+2: interior + top-level renames.
    cleaned: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = _rename_with_table(key, naf_blocks, _NAF_BLOCK_RENAMES)
        new_key = _rename_with_table(new_key, naf_transforms, _NAF_TRANSFORM_RENAMES)
        for old, new in _NAMED_PART_RENAMES.items():
            new_key = new_key.replace(old, new)
        for old, new in _TOPLEVEL_RENAMES.items():
            if new_key.startswith(old):
                new_key = new + new_key[len(old) :]
                break
        cleaned[new_key] = value

    # Pass 3a: per-slice rerooting for the main entropy stack. Discover
    # ``num_slices`` from ``mean_cc_transforms`` first, then drive the rest.
    main_indices = sorted(
        {
            int(key[len("mean_cc_transforms.") :].split(".", 1)[0])
            for key in cleaned
            if key.startswith("mean_cc_transforms.")
        }
    )
    num_slices_main = len(main_indices)

    converted: Dict[str, Tensor] = {}
    consumed: List[str] = []

    if num_slices_main:
        for legacy, container, sub in (
            ("mean_cc_transforms", "latent_codec.y.channel_context", "mean_cc"),
            ("scale_cc_transforms", "latent_codec.y.channel_context", "scale_cc"),
            (
                "mean_support_transforms",
                "latent_codec.y.channel_context",
                "mean_support_transform",
            ),
            (
                "scale_support_transforms",
                "latent_codec.y.channel_context",
                "scale_support_transform",
            ),
            ("lrp_transforms", "latent_codec.y.latent_codec", "lrp_transform"),
        ):
            _reroot_per_slice_keys(
                cleaned,
                converted,
                legacy_prefix=legacy,
                container_prefix=container,
                sub_name=sub,
                num_slices=num_slices_main,
                consume=consumed,
            )
        _replicate_gaussian_conditional(
            cleaned,
            converted,
            legacy_prefix="gaussian_conditional",
            new_prefix="latent_codec.y.latent_codec",
            num_slices=num_slices_main,
            consume=consumed,
        )

    # Pass 3b: per-slice rerooting inside the aux entropy module. Discover
    # ``num_slices_aux`` from ``aux_entropy_model.mean_cc_transforms``.
    aux_indices = sorted(
        {
            int(key[len("aux_entropy_model.mean_cc_transforms.") :].split(".", 1)[0])
            for key in cleaned
            if key.startswith("aux_entropy_model.mean_cc_transforms.")
        }
    )
    num_slices_aux = len(aux_indices)
    if num_slices_aux:
        for legacy, container, sub in (
            (
                "aux_entropy_model.mean_cc_transforms",
                "aux_entropy_model.inner_codec.channel_context",
                "mean_cc",
            ),
            (
                "aux_entropy_model.scale_cc_transforms",
                "aux_entropy_model.inner_codec.channel_context",
                "scale_cc",
            ),
            (
                "aux_entropy_model.mean_support_transforms",
                "aux_entropy_model.inner_codec.channel_context",
                "mean_support_transform",
            ),
            (
                "aux_entropy_model.scale_support_transforms",
                "aux_entropy_model.inner_codec.channel_context",
                "scale_support_transform",
            ),
            (
                "aux_entropy_model.lrp_transforms",
                "aux_entropy_model.inner_codec.latent_codec",
                "lrp_transform",
            ),
        ):
            _reroot_per_slice_keys(
                cleaned,
                converted,
                legacy_prefix=legacy,
                container_prefix=container,
                sub_name=sub,
                num_slices=num_slices_aux,
                consume=consumed,
            )
        _replicate_gaussian_conditional(
            cleaned,
            converted,
            legacy_prefix="aux_entropy_model.gaussian_conditional",
            new_prefix="aux_entropy_model.inner_codec.latent_codec",
            num_slices=num_slices_aux,
            consume=consumed,
        )

    for key in consumed:
        cleaned.pop(key, None)
    # Remaining keys (g_a / g_s / latent_codec.* hyperprior backbone /
    # aux_entropy_model.y_entropy_bottleneck / etc.) pass through unchanged.
    converted.update(cleaned)
    return converted


# ----------------------------------------------------------------------------
# Architecture inference helpers (state_dict -> hyperparameters).
# ----------------------------------------------------------------------------


def _infer_config_from_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, object]:
    """Recover constructor kwargs from a compressai-layout CCA state dict."""
    encoder_dims = (
        state_dict["g_a.down.0.weight"].size(0),
        state_dict["g_a.down.1.weight"].size(0),
        state_dict["g_a.down.2.weight"].size(0),
    )
    latent_channels = state_dict["g_a.down.3.weight"].size(0)
    hyper_channels = state_dict["latent_codec.h_a.4.weight"].size(0)

    encoder_layers: List[int] = []
    for stage in range(3):
        index = 0
        while f"g_a.blocks.{stage}.{index}.beta" in state_dict or _has_resblock(
            state_dict, stage, index
        ):
            index += 1
        encoder_layers.append(index - 3)

    cc_keys = [
        key
        for key in state_dict
        if key.startswith("latent_codec.y.channel_context.y")
        and key.endswith(".mean_cc.4.weight")
    ]
    cc_keys.sort(key=lambda key: int(key.split(".")[3][1:]))  # ".y{k}." -> k
    if not cc_keys:
        raise RuntimeError("state dict does not contain channel-context mean_cc heads")
    slice_sizes = [int(state_dict[key].size(0)) for key in cc_keys]

    em_hidden_channels = int(
        state_dict[
            "latent_codec.y.channel_context.y0.mean_support_transform.input_projection.weight"
        ].size(0)
    )

    em_num_layers = 0
    while (
        f"latent_codec.y.channel_context.y0.mean_support_transform.blocks.{em_num_layers}.beta"
        in state_dict
    ):
        em_num_layers += 1

    cca_training = any(key.startswith("aux_entropy_model.") for key in state_dict)

    return {
        "latent_channels": int(latent_channels),
        "hyper_channels": int(hyper_channels),
        "slice_proportions": tuple(slice_sizes),
        "encoder_dims": tuple(int(value) for value in encoder_dims),
        "encoder_layers": tuple(int(value) for value in encoder_layers),
        "em_hidden_channels": em_hidden_channels,
        "em_num_layers": em_num_layers,
        "cca_training": cca_training,
    }


def _has_resblock(state_dict: Dict[str, Tensor], stage: int, sub_index: int) -> bool:
    return f"g_a.blocks.{stage}.{sub_index}.conv2.weight" in state_dict and (
        f"g_a.blocks.{stage}.{sub_index}.beta" not in state_dict
    )
