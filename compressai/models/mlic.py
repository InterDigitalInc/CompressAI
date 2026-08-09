# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts the MLIC family design from https://github.com/JiangWeibeta/MLIC
# (originally distributed under the Apache License 2.0). Modifications by
# InterDigital Communications, Inc. are released under the BSD 3-Clause Clear
# License terms below.

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

import re

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    EntropyBottleneckLatentCodec,
    HyperpriorLatentCodec,
    MultiContextCheckerboardLatentCodec,
)
from compressai.models._helpers.mlic import (
    AnalysisTransform,
    HyperAnalysis,
    HyperSynthesis,
    LatentResidualPrediction,
    LocalContext,
    StackedCheckerboardConv,
    SynthesisTransform,
    WindowCheckerboardAttn,
)
from compressai.models._helpers.mlicv2 import GSCModule, STMAnalysis, STMSynthesis
from compressai.models._helpers.multi_context_slice import (
    _build_lrp_input_builder,
    _MlicIntraWrapper,
    _MlicppEntropyParameters,
    _MlicppIntraWrapper,
    _MlicppPriorAggregation,
    _MlicppSideLayout,
    _Mlicv2ContextRefinement,
    _Mlicv2HgcpAnchorContext,
    _select_global_inter_factory,
)
from compressai.models.base import CompressionModel
from compressai.registry import register_model

__all__ = [
    "MLIC",
    "MLICPlus",
    "MLICPlusPlus",
    "MLICv2",
]


# MLIC family evolution map:
#
# MLIC  --- + inter-slice global ---> MLIC+
#  |                                   |
#  |   conv stacked checkerboard       |   overlapped window attention
#  |   quadratic global intra          |   quadratic global intra + inter
#  |   N=192, M=320, slice_ch=32       |
#  v                                   v
#  only conv local                 --- replace quadratic with linear attention
#                                      ---> MLIC++
#                                            |
#                    + replace residual blocks with STM
#                    + HGCP: slice-0 global context from hyperprior
#                    + Context Reweighting: channel-wise attention
#                    + 2D RoPE replacing relative position bias
#                    + GSC: post-training skip predictor
#                                            v
#                                          MLICv2


_CURRENT_SLICE_RE = re.compile(r"^latent_codec\.y\.latent_codec\.y(\d+)\.")
_STACKED_CONTEXT_RE = re.compile(
    r"^latent_codec\.y\.latent_codec\.y0\.spatial_context_nonanchor"
    r"\.context\.(\d+)\.weight$"
)


def _infer_slice_num(keys: Iterable[str]) -> int:
    """Infer ``slice_num`` from the per-slice ``latent_codec.y.latent_codec``
    keys of an already-compressai-layout state dict.

    Upstream MLIC++ checkpoints use a different root-level layout; convert them
    first with ``examples/convert_mlic_checkpoint.py`` before calling
    :meth:`from_state_dict`.
    """
    indices: List[int] = []
    for key in keys:
        match = _CURRENT_SLICE_RE.match(key)
        if match is not None:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 10


def _infer_context_window(state_dict: Dict[str, Tensor]) -> int:
    index_key = (
        "latent_codec.y.latent_codec.y0."
        "spatial_context_nonanchor.relative_position_index"
    )
    if index_key in state_dict:
        return int(round(state_dict[index_key].size(0) ** 0.5))

    table_key = (
        "latent_codec.y.latent_codec.y0."
        "spatial_context_nonanchor.relative_position_table"
    )
    if table_key in state_dict:
        side = int(round(state_dict[table_key].size(0) ** 0.5))
        return (side + 1) // 2
    return 5


def _infer_local_kernel(state_dict: Dict[str, Tensor]) -> int:
    key = "latent_codec.y.latent_codec.y0.spatial_context_nonanchor.context.0.weight"
    if key in state_dict:
        return int(state_dict[key].size(-1))
    return 5


def _infer_local_layers(keys: Iterable[str]) -> int:
    indices = []
    for key in keys:
        match = _STACKED_CONTEXT_RE.match(key)
        if match is not None:
            indices.append(int(match.group(1)))
    return len(indices) if indices else 3


class _SideContextChannelGroupsLatentCodec(ChannelGroupsLatentCodec):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if "y0" not in self.channel_context:
            raise ValueError("side-parameter channel groups require channel_context.y0")

    def _get_ctx_params(
        self, k: int, side_params: Tensor, y_hat_: List[Tensor]
    ) -> Tensor:
        if k == 0:
            return self.channel_context["y0"](side_params)
        support = [y_hat_[i] for i in self.support_slices[k]]
        if not support:
            return self.channel_context[f"y{k}"](side_params)
        return self.channel_context[f"y{k}"](
            self.merge_params(side_params, self.merge_y(*support))
        )


class _BaseMLIC(CompressionModel):
    _variant = "mlic"
    _analysis_cls = AnalysisTransform
    _synthesis_cls = SynthesisTransform

    def __init__(
        self,
        *,
        N: int,
        M: int,
        slice_num: int,
        context_window: int = 5,
        local_kernel: int = 5,
        local_layers: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if slice_num <= 0:
            raise ValueError("slice_num must be positive")
        if context_window % 2 == 0:
            raise ValueError("context_window must be odd")
        if M % slice_num != 0:
            raise ValueError("M must be divisible by slice_num")

        self.N = int(N)
        self.M = int(M)
        self.context_window = int(context_window)
        self.local_kernel = int(local_kernel)
        self.local_layers = int(local_layers)
        self.slice_num = int(slice_num)
        self.slice_ch = int(M // slice_num)

        self.g_a = self._analysis_cls(N=N, M=M)
        self.g_s = self._synthesis_cls(N=N, M=M)

        # Per-slice entropy stack, inlined ELIC-style. The variant only changes
        # three things: the inter-slice global context factory, the non-anchor
        # spatial context module, and the intra-channel context wrapper (plus
        # the MLICv2-only HGCP anchor context and GSC selective predictor).
        variant = self._variant
        slice_ch = self.slice_ch
        global_inter_factory = _select_global_inter_factory(variant)
        use_global_inter = global_inter_factory is not None

        def _side_layout(k: int) -> _MlicppSideLayout:
            return _MlicppSideLayout(
                M=M,
                slice_ch=slice_ch,
                slice_index=k,
                use_global_inter=use_global_inter,
            )

        def _spatial_context() -> nn.Module:
            if variant == "mlic":
                return StackedCheckerboardConv(
                    dim=slice_ch,
                    kernel=local_kernel,
                    num_layers=local_layers,
                )
            if variant == "mlic+":
                return WindowCheckerboardAttn(dim=slice_ch, window_size=context_window)
            return LocalContext(dim=slice_ch, window_size=context_window)

        def _intra_context(layout: _MlicppSideLayout) -> nn.Module:
            if variant in ("mlic", "mlic+"):
                return _MlicIntraWrapper(layout)
            context = _MlicppIntraWrapper(layout)
            if variant == "mlicv2":
                return _Mlicv2ContextRefinement(context, dim=2 * slice_ch)
            return context

        def _leaf(k: int) -> MultiContextCheckerboardLatentCodec:
            layout = _side_layout(k)
            anchor_context_ch = 2 * slice_ch if variant == "mlicv2" and k == 0 else 0
            return MultiContextCheckerboardLatentCodec(
                entropy_parameters_anchor=_MlicppEntropyParameters(
                    layout,
                    step="anchor",
                    anchor_context_ch=anchor_context_ch,
                ),
                entropy_parameters_nonanchor=_MlicppEntropyParameters(
                    layout,
                    step="non_anchor",
                ),
                spatial_context_anchor=(
                    _Mlicv2HgcpAnchorContext(M=M, slice_ch=slice_ch)
                    if variant == "mlicv2" and k == 0
                    else None
                ),
                spatial_context_nonanchor=_spatial_context(),
                intra_channel_context_nonanchor=(
                    _intra_context(layout) if k > 0 else None
                ),
                selective_predictor=(
                    GSCModule(slice_ch=slice_ch, side_ch=layout.side_ch)
                    if variant == "mlicv2"
                    else None
                ),
                lrp_anchor=LatentResidualPrediction(
                    in_dim=M + (k + 1) * slice_ch,
                    out_dim=slice_ch,
                ),
                lrp_nonanchor=LatentResidualPrediction(
                    in_dim=M + (k + 1) * slice_ch,
                    out_dim=slice_ch,
                ),
                lrp_input_builder=_build_lrp_input_builder(layout),
                lrp_activation=None,
                lrp_scale=1.0,
                anchor_parity="odd",
            )

        support_slices = [list(range(k)) for k in range(slice_num)]

        self.latent_codec = HyperpriorLatentCodec(
            h_a=HyperAnalysis(M=M, N=N),
            h_s=HyperSynthesis(M=M, N=N),
            latent_codec={
                "z": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    quantizer="ste",
                ),
                "y": _SideContextChannelGroupsLatentCodec(
                    groups=[slice_ch] * slice_num,
                    channel_context={
                        f"y{k}": _MlicppPriorAggregation(
                            M=M,
                            slice_ch=slice_ch,
                            slice_index=k,
                            global_inter_factory=global_inter_factory,
                        )
                        for k in range(slice_num)
                    },
                    latent_codec={f"y{k}": _leaf(k) for k in range(slice_num)},
                    support_slices=support_slices,
                ),
            },
        )

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        return {
            "x_hat": self.g_s(y_out["y_hat"]),
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        y_out = self.latent_codec.compress(y)
        return {"strings": y_out["strings"], "shape": y_out["shape"]}

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Dict[str, Union[List[Tuple[int, ...]], Tuple[int, ...]]],
    ) -> Dict[str, Tensor]:
        y_out = self.latent_codec.decompress(strings, shape)
        return {"x_hat": self.g_s(y_out["y_hat"])}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "_BaseMLIC":
        N = state_dict["g_a.analysis_transform.0.conv1.weight"].size(0)
        M = state_dict["g_a.analysis_transform.6.weight"].size(0)
        slice_num = _infer_slice_num(state_dict.keys())
        kwargs = {
            "N": N,
            "M": M,
            "slice_num": slice_num,
        }
        if cls._variant in ("mlic+", "mlicpp", "mlicv2"):
            kwargs["context_window"] = _infer_context_window(state_dict)
        else:
            kwargs["local_kernel"] = _infer_local_kernel(state_dict)
            kwargs["local_layers"] = _infer_local_layers(state_dict.keys())

        net = cls(**kwargs)
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            key for key in net.state_dict() if key.endswith("relative_position_index")
        }
        missing_keys = set(incompatible_keys.missing_keys) - allowed_missing
        if missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                f"Unexpected incompatibility while loading {cls.__name__} state_dict: "
                f"missing={sorted(missing_keys)}, "
                f"unexpected={sorted(incompatible_keys.unexpected_keys)}"
            )
        return net


@register_model("mlic")
class MLIC(_BaseMLIC):
    r"""MLIC model from W. Jiang, J. Yang, Y. Zhai, P. Ning, F. Gao, R. Wang:
    `"MLIC: Multi-Reference Entropy Model for Learned Image Compression"
    <https://arxiv.org/abs/2211.07273>`_, ACM Multimedia 2023.
    """

    _variant = "mlic"

    def __init__(
        self,
        N: int = 192,
        M: int = 192,
        slice_num: int = 6,
        local_kernel: int = 5,
        local_layers: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            N=N,
            M=M,
            slice_num=slice_num,
            local_kernel=local_kernel,
            local_layers=local_layers,
            **kwargs,
        )


@register_model("mlicplus")
class MLICPlus(_BaseMLIC):
    r"""MLIC+ model from W. Jiang, J. Yang, Y. Zhai, P. Ning, F. Gao, R. Wang:
    `"MLIC: Multi-Reference Entropy Model for Learned Image Compression"
    <https://arxiv.org/abs/2211.07273>`_, ACM Multimedia 2023.
    """

    _variant = "mlic+"

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        slice_num: int = 10,
        context_window: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            N=N,
            M=M,
            slice_num=slice_num,
            context_window=context_window,
            **kwargs,
        )


@register_model("mlicpp")
class MLICPlusPlus(_BaseMLIC):
    r"""MLIC++ model from W. Jiang, J. Yang, Y. Zhai, F. Gao, R. Wang:
    `"MLIC++: Linear Complexity Multi-Reference Entropy Modeling for Learned
    Image Compression" <https://arxiv.org/abs/2307.15421>`_, ACM Trans.
    Multimedia Comput. Commun. Appl. (TOMM), 2025; ICML 2023 Neural
    Compression Workshop.

    This implementation uses a containerized hyperprior entropy stack:
    ``HyperpriorLatentCodec`` wraps the MLIC++ hyper transforms and a
    ``ChannelGroupsLatentCodec`` built from per-slice
    ``MultiContextCheckerboardLatentCodec`` leaves.

    Upstream MLIC++ checkpoints from JiangWeibeta/MLIC use a different
    root-level key layout; convert them to the compressai layout first with
    ``examples/convert_mlic_checkpoint.py`` before calling
    :meth:`from_state_dict`.
    """

    _variant = "mlicpp"

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        slice_num: int = 10,
        context_window: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            N=N,
            M=M,
            slice_num=slice_num,
            context_window=context_window,
            **kwargs,
        )


@register_model("mlicv2")
class MLICv2(_BaseMLIC):
    r"""MLICv2 model from W. Jiang, J. Yang, Y. Zhai, F. Gao, R. Wang:
    `"MLIC++: Linear Complexity Multi-Reference Entropy Modeling for Learned
    Image Compression" <https://arxiv.org/abs/2307.15421>`_ follow-up family.

    This variant replaces the MLIC++ analysis/synthesis transforms with STM
    blocks and enables HGCP, context reweighting, 2D RoPE, and GSC in the
    shared multi-context slice factory.
    """

    _variant = "mlicv2"
    _analysis_cls = STMAnalysis
    _synthesis_cls = STMSynthesis

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        slice_num: int = 10,
        context_window: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            N=N,
            M=M,
            slice_num=slice_num,
            context_window=context_window,
            **kwargs,
        )
