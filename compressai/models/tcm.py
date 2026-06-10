# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/jmliu206/LIC_TCM
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

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from compressai.layers import (
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.layers.attn import ConvTransBlock, SWAtten
from compressai.models._helpers.channel_context import MeanScaleContextHead
from compressai.models._helpers.slice_helpers import (
    infer_max_support_slices,
    infer_num_slices,
    make_entropy_transform,
)
from compressai.models.base import SimpleVAECompressionModel
from compressai.registry import register_model

__all__ = [
    "TCM",
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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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


# ----------------------------------------------------------------------------
# Architecture inference helpers (state_dict -> hyperparameters)
# ----------------------------------------------------------------------------


def _group_consecutive(indices: Iterable[int]) -> List[List[int]]:
    grouped: List[List[int]] = []
    for index in sorted(indices):
        if not grouped or index != grouped[-1][-1] + 1:
            grouped.append([index])
            continue
        grouped[-1].append(index)
    return grouped


def _infer_stage_groups(state_dict: Dict[str, Tensor], prefix: str) -> List[List[int]]:
    indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith(f"{prefix}.") and ".conv1_1.weight" in key
    }
    return _group_consecutive(indices)


def _infer_stage_depths(state_dict: Dict[str, Tensor]) -> Optional[List[int]]:
    g_a_groups = _infer_stage_groups(state_dict, "g_a")
    g_s_groups = _infer_stage_groups(state_dict, "g_s")
    if len(g_a_groups) != 3 or len(g_s_groups) != 3:
        return None
    return [len(group) for group in g_a_groups + g_s_groups]


def _infer_head_dims(state_dict: Dict[str, Tensor], N: int) -> Optional[List[int]]:
    head_dims: List[int] = []
    for prefix in ("g_a", "g_s"):
        for group in _infer_stage_groups(state_dict, prefix):
            if not group:
                continue
            table_key = (
                f"{prefix}.{group[0]}.trans_block.msa.attn.relative_position_bias_table"
            )
            if table_key not in state_dict:
                return None
            num_heads = state_dict[table_key].size(1)
            head_dims.append(N // num_heads)
    return head_dims if len(head_dims) == 6 else None


def _infer_hyper_head_dim(state_dict: Dict[str, Tensor], N: int, default: int) -> int:
    for key in (
        "h_a.1.trans_block.msa.attn.relative_position_bias_table",
        "h_mean_s.1.trans_block.msa.attn.relative_position_bias_table",
    ):
        if key in state_dict:
            return N // state_dict[key].size(1)
    return default


# ----------------------------------------------------------------------------
# Architecture building blocks
# ----------------------------------------------------------------------------


def _make_mixed_stage(
    depth: int,
    branch_channels: int,
    head_dim: int,
    window_size: int,
    drop_paths: Sequence[float],
    tail: nn.Module,
) -> List[nn.Module]:
    if len(drop_paths) != depth:
        raise ValueError("drop_paths must match stage depth")
    blocks = [
        ConvTransBlock(
            branch_channels,
            branch_channels,
            head_dim,
            window_size,
            drop_paths[index],
            type="W" if index % 2 == 0 else "SW",
        )
        for index in range(depth)
    ]
    return [*blocks, tail]


# ----------------------------------------------------------------------------
# TCM model
# ----------------------------------------------------------------------------


@register_model("lic-tcm")
@register_model("tcm")
class TCM(SimpleVAECompressionModel):
    r"""TCM model from J. Liu, H. Sun, J. Katto: `"Learned Image Compression
    with Mixed Transformer-CNN Architectures"
    <https://arxiv.org/abs/2303.14978>`_, IEEE/CVF Conf. on Computer Vision
    and Pattern Recognition (CVPR), 2023 (Highlight).

    Stacks parallel Transformer-CNN Mixture (TCM) blocks for the
    analysis/synthesis transforms and uses a channel-wise autoregressive
    entropy model with parameter-efficient swin-transformer attention
    (``SWAtten``).

    The entropy stack is a fully containerised
    :class:`HyperpriorLatentCodec` that owns ``h_a``, ``h_s``, the ``z``
    bottleneck and the per-slice side-conditioned channel-groups path. The
    channel-context heads route per-slice ``mean_in`` / ``scale_in`` through
    independent SWAtten instances before the 3-conv ``mean_cc`` /
    ``scale_cc`` stacks (TCM's distinctive widths ``(224, 128)``).

    Args:
        N (int): Channel width of the analysis/synthesis transform branches.
        M (int): Channels in the latent representation ``y``.
        hyper_channels (int): Channels in the hyperprior backbone ``z``.
        num_slices (int): Number of channel slices for the entropy model.
        max_support_slices (int): Per-slice context cap.
    """

    def __init__(
        self,
        config: Optional[Sequence[int]] = None,
        head_dim: Optional[Sequence[int]] = None,
        drop_path_rate: float = 0.0,
        N: int = 128,
        M: int = 320,
        hyper_channels: int = 192,
        num_slices: int = 5,
        max_support_slices: int = 5,
        window_size: int = 8,
        hyper_window_size: int = 4,
        hyper_head_dim: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        config = tuple(int(value) for value in (config or (2, 2, 2, 2, 2, 2)))
        head_dim = tuple(int(value) for value in (head_dim or (8, 16, 32, 32, 16, 8)))
        if len(config) != 6:
            raise ValueError("config must provide six stage depths")
        if len(head_dim) != 6:
            raise ValueError("head_dim must provide six stage head dimensions")
        if any(value < 0 for value in config):
            raise ValueError("config values must be non-negative")
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")
        if any(N % value != 0 for value in head_dim):
            raise ValueError("Each head_dim must divide N")
        if N % hyper_head_dim != 0:
            raise ValueError("hyper_head_dim must divide N")

        self.config = config
        self.head_dim = head_dim
        self.window_size = int(window_size)
        self.hyper_window_size = int(hyper_window_size)
        self.hyper_head_dim = int(hyper_head_dim)
        self.N = int(N)
        self.M = int(M)
        self.hyper_channels = int(hyper_channels)
        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)

        drop_paths = torch.linspace(0, drop_path_rate, sum(config)).tolist()
        offset = 0

        def stage_drop_paths(depth: int) -> List[float]:
            nonlocal offset
            values = [float(value) for value in drop_paths[offset : offset + depth]]
            offset += depth
            return values

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, 2 * N, stride=2),
            *_make_mixed_stage(
                config[0],
                N,
                head_dim[0],
                self.window_size,
                stage_drop_paths(config[0]),
                ResidualBlockWithStride(2 * N, 2 * N, stride=2),
            ),
            *_make_mixed_stage(
                config[1],
                N,
                head_dim[1],
                self.window_size,
                stage_drop_paths(config[1]),
                ResidualBlockWithStride(2 * N, 2 * N, stride=2),
            ),
            *_make_mixed_stage(
                config[2],
                N,
                head_dim[2],
                self.window_size,
                stage_drop_paths(config[2]),
                conv3x3(2 * N, M, stride=2),
            ),
        )
        self.g_s = nn.Sequential(
            ResidualBlockUpsample(M, 2 * N, 2),
            *_make_mixed_stage(
                config[3],
                N,
                head_dim[3],
                self.window_size,
                stage_drop_paths(config[3]),
                ResidualBlockUpsample(2 * N, 2 * N, 2),
            ),
            *_make_mixed_stage(
                config[4],
                N,
                head_dim[4],
                self.window_size,
                stage_drop_paths(config[4]),
                ResidualBlockUpsample(2 * N, 2 * N, 2),
            ),
            *_make_mixed_stage(
                config[5],
                N,
                head_dim[5],
                self.window_size,
                stage_drop_paths(config[5]),
                subpel_conv3x3(2 * N, 3, 2),
            ),
        )

        h_a = nn.Sequential(
            ResidualBlockWithStride(M, 2 * N, 2),
            *_make_mixed_stage(
                config[0],
                N,
                self.hyper_head_dim,
                self.hyper_window_size,
                [0.0] * config[0],
                conv3x3(2 * N, hyper_channels, stride=2),
            ),
        )
        h_mean_s = nn.Sequential(
            ResidualBlockUpsample(hyper_channels, 2 * N, 2),
            *_make_mixed_stage(
                config[3],
                N,
                self.hyper_head_dim,
                self.hyper_window_size,
                [0.0] * config[3],
                subpel_conv3x3(2 * N, M, 2),
            ),
        )
        h_scale_s = nn.Sequential(
            ResidualBlockUpsample(hyper_channels, 2 * N, 2),
            *_make_mixed_stage(
                config[3],
                N,
                self.hyper_head_dim,
                self.hyper_window_size,
                [0.0] * config[3],
                subpel_conv3x3(2 * N, M, 2),
            ),
        )

        slice_ch = M // num_slices
        widths = (224, 128)
        groups = [slice_ch] * num_slices
        window_size = self.window_size

        def support_count(k: int) -> int:
            return k if max_support_slices < 0 else min(k, max_support_slices)

        def mean_support_ch(k: int) -> int:
            return M + slice_ch * support_count(k)

        def swatten_factory(c_in: int, c_out: int) -> nn.Module:
            # Independent SWAtten per mean / scale path, mirroring upstream
            # atten_mean[k] / atten_scale[k].
            return SWAtten(
                input_dim=c_in,
                output_dim=c_out,
                head_dim=16,
                window_size=window_size,
                drop_path=0.0,
                inter_dim=128,
            )

        # Side-parameter channel-groups wiring, inlined ELIC-style. Differs from
        # WACNN/STF only in widths=(224, 128) and the per-slice SWAtten
        # support transforms wrapping mean_in / scale_in.
        channel_context = {
            f"y{k}": MeanScaleContextHead(
                mean_cc=make_entropy_transform(
                    mean_support_ch(k), slice_ch, widths=widths
                ),
                scale_cc=make_entropy_transform(
                    mean_support_ch(k), slice_ch, widths=widths
                ),
                mean_support_transform=swatten_factory(
                    mean_support_ch(k), mean_support_ch(k)
                ),
                scale_support_transform=swatten_factory(
                    mean_support_ch(k), mean_support_ch(k)
                ),
                side_split=M,
                emit_mean_support=True,
            )
            for k in range(num_slices)
        }
        y_latent_codec = {
            f"y{k}": _LRPGaussianLatentCodec(
                lrp_transform=make_entropy_transform(
                    mean_support_ch(k) + slice_ch, slice_ch, widths=widths
                ),
                mean_support_trail_channels=mean_support_ch(k),
                quantizer="ste",
            )
            for k in range(num_slices)
        }

        self.latent_codec = HyperpriorLatentCodec(
            h_a=h_a,
            h_s=_DualHyperSynthesis(h_mean_s, h_scale_s),
            latent_codec={
                "z": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(hyper_channels),
                    quantizer="ste",
                ),
                "y": _SideContextChannelGroupsLatentCodec(
                    groups=groups,
                    channel_context=channel_context,
                    latent_codec=y_latent_codec,
                    max_support_slices=max_support_slices,
                ),
            },
        )

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "TCM":
        N = state_dict["g_a.0.conv1.weight"].size(0) // 2
        M = state_dict["latent_codec.h_a.0.conv1.weight"].size(1)
        config = _infer_stage_depths(state_dict) or [2, 2, 2, 2, 2, 2]
        head_dim = _infer_head_dims(state_dict, N) or [8, 16, 32, 32, 16, 8]
        hyper_channels = state_dict["latent_codec.z.entropy_bottleneck.quantiles"].size(
            0
        )
        num_slices = infer_num_slices(state_dict) or 5
        max_support_slices = infer_max_support_slices(state_dict, M, num_slices)
        net = cls(
            config=config,
            head_dim=head_dim,
            N=N,
            M=M,
            hyper_channels=hyper_channels,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            hyper_head_dim=_infer_hyper_head_dim(state_dict, N, 32),
        )
        # ConvTransBlock's WindowAttention registers
        # ``relative_position_index`` as a non-persistent buffer, so it is
        # absent from saved state dicts. Tolerate the missing keys.
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            key for key in net.state_dict() if key.endswith("relative_position_index")
        }
        missing_keys = set(incompatible_keys.missing_keys) - allowed_missing
        if missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                "Unexpected incompatibility while loading TCM state_dict: "
                f"missing={sorted(missing_keys)}, "
                f"unexpected={sorted(incompatible_keys.unexpected_keys)}"
            )
        return net
