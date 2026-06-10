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

The main entropy stack is a fully containerised
:class:`HyperpriorLatentCodec` with variable-length slice groups and
per-slice :class:`_NAFTransform` support transforms. The optional auxiliary
CCA branch (:class:`_CCAAuxEntropyModel`) is a separate ``nn.Module`` that
re-encodes ``y`` with the skip-most-recent support selection used by
:class:`compressai.losses.CCARateDistortionLoss` to align the causal
context with the rate-distortion objective.
"""

from __future__ import annotations

import math

from itertools import accumulate
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    EntropyBottleneckLatentCodec,
    GaussianConditionalLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers.layers import conv1x1
from compressai.models._helpers.channel_context import MeanScaleContextHead
from compressai.models._helpers.slice_helpers import make_entropy_transform
from compressai.models.base import CompressionModel, get_scale_table
from compressai.models.sensetime import ResidualBottleneckBlock
from compressai.models.utils import conv, deconv
from compressai.registry import register_model

__all__ = [
    "CCAModel",
]


class _DualHyperSynthesis(nn.Module):
    h_mean_s: nn.Module
    h_scale_s: nn.Module

    def __init__(self, h_mean_s: nn.Module, h_scale_s: nn.Module) -> None:
        super().__init__()
        self.h_mean_s = h_mean_s
        self.h_scale_s = h_scale_s

    def forward(self, z_hat: Tensor) -> Tensor:
        return torch.cat([self.h_mean_s(z_hat), self.h_scale_s(z_hat)], dim=1)


class _LRPGaussianLatentCodec(GaussianConditionalLatentCodec):
    lrp_transform: nn.Module

    def __init__(
        self,
        lrp_transform: nn.Module,
        *,
        lrp_scale: float = 0.5,
        mean_support_trail_channels: int = 0,
        **gc_kwargs: Any,
    ) -> None:
        super().__init__(**gc_kwargs)
        self.lrp_transform = lrp_transform
        self.lrp_scale = float(lrp_scale)
        self.mean_support_trail_channels = int(mean_support_trail_channels)

    def _split_ctx_params(self, ctx_params: Tensor) -> Tuple[Tensor, Tensor]:
        if self.mean_support_trail_channels <= 0:
            return ctx_params, ctx_params
        trail = self.mean_support_trail_channels
        gaussian_params = ctx_params[:, :-trail]
        mean_support = ctx_params[:, -trail:]
        return gaussian_params, mean_support

    def _apply_lrp(self, mean_support: Tensor, y_hat: Tensor) -> Tensor:
        lrp = self.lrp_scale * torch.tanh(
            self.lrp_transform(torch.cat([mean_support, y_hat], dim=1))
        )
        return y_hat + lrp

    def forward(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params, mean_support = self._split_ctx_params(ctx_params)
        out = super().forward(y, gaussian_params)
        out["y_hat"] = self._apply_lrp(mean_support, out["y_hat"])
        return out

    def compress(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params, mean_support = self._split_ctx_params(ctx_params)
        out = super().compress(y, gaussian_params)
        out["y_hat"] = self._apply_lrp(mean_support, out["y_hat"])
        return out

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, ...],
        ctx_params: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        gaussian_params, mean_support = self._split_ctx_params(ctx_params)
        out = super().decompress(strings, shape, gaussian_params, **kwargs)
        out["y_hat"] = self._apply_lrp(mean_support, out["y_hat"])
        return out


class _SideContextChannelGroupsLatentCodec(ChannelGroupsLatentCodec):
    def __init__(
        self,
        *args,
        support_filter: Optional[Callable[[int, List[Tensor]], List[Tensor]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.support_filter = support_filter
        if "y0" not in self.channel_context:
            raise ValueError("side-parameter channel groups require channel_context.y0")

    def _get_ctx_params(
        self, k: int, side_params: Tensor, y_hat_: List[Tensor]
    ) -> Tensor:
        if k == 0:
            return self.channel_context["y0"](side_params)
        support = self._select_support(k, y_hat_)
        if not support:
            return self.channel_context[f"y{k}"](side_params)
        return self.channel_context[f"y{k}"](
            self.merge_params(side_params, self.merge_y(*support))
        )

    def _select_support(self, k: int, y_hat_: List[Tensor]) -> List[Tensor]:
        prior = list(y_hat_[:k])
        if self.support_filter is not None:
            return list(self.support_filter(k, prior))
        return super()._select_support(k, y_hat_)


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
    after ``convert_upstream_cca_state_dict`` (in
    ``examples/convert_cca_checkpoint.py``) so released checkpoints load 1:1.
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
# Auxiliary CCA entropy branch.
# ----------------------------------------------------------------------------


class _CCAAuxEntropyModel(nn.Module):
    """Auxiliary CCA entropy branch (skip-most-recent-slice support).

    Produces the ``y_aux`` (factorised) and ``y_cca`` (Gaussian-conditional)
    likelihoods used by :class:`compressai.losses.CCARateDistortionLoss`.

    Mirrors the upstream ``AuxEntropyModel`` in
    ``candidate/CCA/models/aux_em.py``: for slice ``i`` the support is
    ``cat(latent_means, *y_hat_slices[: max(i - 1, 0)])`` (i.e., skip the
    *most recent* decoded slice). This is wired inline (ELIC-style) on a
    private side-parameter channel-groups codec with matching per-slice
    ``support_count`` to size the channel-context heads.

    Although upstream only *uses* the LRP path on the first ``num_slices -
    2`` slices, the published checkpoints carry LRP weights for *all*
    slices. To strict-load those checkpoints every leaf is a Gaussian codec
    with local LRP refinement; the LRP applied to the trailing two slices is
    benign (those slices' ``y_hat`` is excluded from every later slice's
    skip-most-recent support selection, so it never feeds back into the
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

        M = self.latent_channels
        slice_sizes = self.slice_sizes
        em_hidden_channels = self.hidden_channels
        em_num_layers = self.num_layers
        cumulative = list(accumulate(slice_sizes, initial=0))
        widths = (em_hidden_channels, 128)

        # Skip-most-recent support: slice k sees max(k - 1, 0) prior slices.
        def support_count(k: int) -> int:
            return max(k - 1, 0)

        def support_filter(k: int, prior: List[Tensor]) -> List[Tensor]:
            return prior[: max(k - 1, 0)]

        def mean_support_ch(k: int) -> int:
            return M + cumulative[support_count(k)]

        def naf_factory(c_in: int, c_out: int) -> nn.Module:
            return _NAFTransform(c_in, c_out, em_hidden_channels, em_num_layers)

        # Side-parameter channel-groups wiring, inlined ELIC-style. Differs
        # from the main CCA stack only in the skip-most-recent support count.
        channel_context = {
            f"y{k}": MeanScaleContextHead(
                mean_cc=make_entropy_transform(
                    mean_support_ch(k), slice_sizes[k], widths=widths
                ),
                scale_cc=make_entropy_transform(
                    mean_support_ch(k), slice_sizes[k], widths=widths
                ),
                mean_support_transform=naf_factory(
                    mean_support_ch(k), mean_support_ch(k)
                ),
                scale_support_transform=naf_factory(
                    mean_support_ch(k), mean_support_ch(k)
                ),
                side_split=M,
                emit_mean_support="post",
            )
            for k in range(self.num_slices)
        }
        y_latent_codec = {
            f"y{k}": _LRPGaussianLatentCodec(
                lrp_transform=make_entropy_transform(
                    mean_support_ch(k) + slice_sizes[k], slice_sizes[k], widths=widths
                ),
                mean_support_trail_channels=mean_support_ch(k),
                quantizer="ste",
            )
            for k in range(self.num_slices)
        }

        self.y_entropy_bottleneck = EntropyBottleneck(M)
        self.inner_codec = _SideContextChannelGroupsLatentCodec(
            groups=list(slice_sizes),
            channel_context=channel_context,
            latent_codec=y_latent_codec,
            max_support_slices=-1,
            support_filter=support_filter,
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

    The entropy stack is a :class:`HyperpriorLatentCodec` with variable-length
    channel groups (``slice_proportions``), per-slice :class:`_NAFTransform`
    support transforms, and a STE-quantised ``z`` leaf. When
    ``cca_training=True`` an auxiliary
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

        # Main entropy stack, wired inline (ELIC-style). Distinctive choices
        # vs. STF/WACNN/TCM:
        #
        # - ``groups`` is the variable-length ``slice_sizes`` (resolved from
        #   ``slice_proportions``); STF / WACNN / TCM use uniform ``[M//K]*K``.
        # - the per-slice mean / scale support transforms are _NAFTransform
        #   instances (vs. STF identity / TCM SWAtten), with
        #   ``emit_mean_support="post"`` so the LRP head receives the
        #   *post*-NAFTransform mean support — replicating the upstream LIC
        #   LRP layout for byte-for-byte weight transfer.
        # - the ``z`` leaf uses ``EntropyBottleneckLatentCodec(quantizer="ste")``
        #   to recover upstream's ``quantize_ste(z - z_offset) + z_offset``
        #   behaviour without a model-side hack.
        M = self.M
        slice_sizes = self.slice_sizes
        cumulative = list(accumulate(slice_sizes, initial=0))
        widths = (self.em_hidden_channels, 128)

        # use-all-prior support; matches upstream LIC main path (no skip).
        def mean_support_ch(k: int) -> int:
            # cat(latent_means(M), *prev_y_hat(sum(slice_sizes[:k]))).
            return M + cumulative[k]

        def naf_factory(c_in: int, c_out: int) -> nn.Module:
            return _NAFTransform(
                c_in, c_out, self.em_hidden_channels, self.em_num_layers
            )

        channel_context = {
            f"y{k}": MeanScaleContextHead(
                mean_cc=make_entropy_transform(
                    mean_support_ch(k), slice_sizes[k], widths=widths
                ),
                scale_cc=make_entropy_transform(
                    mean_support_ch(k), slice_sizes[k], widths=widths
                ),
                mean_support_transform=naf_factory(
                    mean_support_ch(k), mean_support_ch(k)
                ),
                scale_support_transform=naf_factory(
                    mean_support_ch(k), mean_support_ch(k)
                ),
                side_split=M,
                emit_mean_support="post",
            )
            for k in range(self.num_slices)
        }
        y_latent_codec = {
            f"y{k}": _LRPGaussianLatentCodec(
                lrp_transform=make_entropy_transform(
                    mean_support_ch(k) + slice_sizes[k], slice_sizes[k], widths=widths
                ),
                mean_support_trail_channels=mean_support_ch(k),
                quantizer="ste",
            )
            for k in range(self.num_slices)
        }

        self.latent_codec = HyperpriorLatentCodec(
            h_a=h_a,
            h_s=_DualHyperSynthesis(h_mean_s, h_scale_s),
            latent_codec={
                "z": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(self.N),
                    quantizer="ste",
                ),
                "y": _SideContextChannelGroupsLatentCodec(
                    groups=list(slice_sizes),
                    channel_context=channel_context,
                    latent_codec=y_latent_codec,
                    max_support_slices=-1,
                ),
            },
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
            # ``self.latent_codec.h_s`` concatenates the dual hyper-synthesis heads; its
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
        shape: Dict[str, Tuple[int, ...]],
    ) -> Dict[str, Tensor]:
        y_out = self.latent_codec.decompress(strings, shape)
        return {"x_hat": self.g_s(y_out["y_hat"]).clamp_(0, 1)}

    def update(
        self, scale_table: Optional[Tensor] = None, force: bool = False, **kwargs
    ) -> bool:
        if scale_table is None:
            scale_table = get_scale_table()
        return super().update(scale_table=scale_table, force=force, **kwargs)

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "CCAModel":
        cfg = _infer_config_from_state_dict(state_dict)
        net = cls(**cfg)
        net.load_state_dict(state_dict)
        return net


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
