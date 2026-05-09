# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/Googolxx/STF
# (originally distributed under the Apache License 2.0). The upstream copyright
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

import math
import re

from typing import Dict, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn

from timm.layers import DropPath, Mlp
from timm.models.swin_transformer import SwinTransformerBlock as _TimmSwinBlock
from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    DualHyperSynthesis,
    EntropyBottleneckLatentCodec,
    HyperpriorLatentCodec,
    LRPGaussianLatentCodec,
)
from compressai.latent_codecs._slice_helpers import (
    infer_max_support_slices,
    infer_num_slices,
    make_entropy_transform,
)
from compressai.layers import GDN, conv3x3, subpel_conv3x3
from compressai.layers.attn import (
    PatchMerging,
    PatchSplit,
    WinNoShiftAttention,
)
from compressai.models._helpers.channel_context import build_mean_scale_head
from compressai.models._helpers.channel_slice import build_channel_slice_codec
from compressai.models.base import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.registry import register_model

__all__ = [
    "SymmetricalTransFormer",
    "WACNN",
    "convert_upstream_stf_state_dict",
]


# ----------------------------------------------------------------------------
# STF building blocks
# (formerly compressai/layers/lic/stf.py; private to the WACNN / SymmetricalTransFormer models)
# ----------------------------------------------------------------------------


class _STFBasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        downsample: Optional[Type[nn.Module]] = None,
    ) -> None:
        del qk_scale  # timm SwinTransformerBlock derives scale from head_dim
        super().__init__()
        drop_path_values = (
            list(drop_path)
            if isinstance(drop_path, Sequence)
            and not isinstance(drop_path, (str, bytes))
            else [float(drop_path)] * depth
        )
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList(
            [
                _TimmSwinBlock(
                    dim=dim,
                    input_resolution=(0, 0),  # ignored when always_partition=True
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_values[index],
                    norm_layer=norm_layer,
                    always_partition=True,  # keep configured window/shift even if input is small
                    dynamic_mask=True,
                )
                for index in range(depth)
            ]
        )
        self.downsample = (
            downsample(dim=dim, norm_layer=norm_layer) if downsample else None
        )

        # Released STF checkpoints carry `attn.relative_position_index` per block
        # (the upstream WindowAttention registers it as a persistent buffer).
        # timm's WindowAttention uses persistent=False, so promote it here so
        # strict-mode state_dict loading round-trips without filtering keys.
        for block in self.blocks:
            index = block.attn.relative_position_index
            del block.attn._buffers["relative_position_index"]
            block.attn.register_buffer(
                "relative_position_index", index, persistent=True
            )

    def forward(
        self, input_tensor: Tensor, height: int, width: int
    ) -> tuple[Tensor, int, int]:
        batch_size, length, channels = input_tensor.shape
        if length != height * width:
            raise ValueError("input feature has wrong size")
        x = input_tensor.view(batch_size, height, width, channels)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(batch_size, height * width, channels)

        if self.downsample is None:
            return x, height, width

        x = self.downsample(x, height, width)
        if isinstance(self.downsample, PatchMerging):
            return x, (height + 1) // 2, (width + 1) // 2
        return x, height * 2, width * 2


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, input_tensor: Tensor) -> Tensor:
        _, _, height, width = input_tensor.size()
        if width % self.patch_size[1] != 0:
            input_tensor = nn.functional.pad(
                input_tensor,
                (0, self.patch_size[1] - width % self.patch_size[1]),
            )
        if height % self.patch_size[0] != 0:
            input_tensor = nn.functional.pad(
                input_tensor,
                (0, 0, 0, self.patch_size[0] - height % self.patch_size[0]),
            )

        output = self.proj(input_tensor)
        if self.norm is None:
            return output

        out_height, out_width = output.size(2), output.size(3)
        output = output.flatten(2).transpose(1, 2)
        output = self.norm(output)
        return output.transpose(1, 2).view(-1, self.embed_dim, out_height, out_width)


# ----------------------------------------------------------------------------
# STF / WACNN models
# ----------------------------------------------------------------------------


_UPSTREAM_LATENT_CODEC_PREFIXES = (
    "cc_mean_transforms",
    "cc_scale_transforms",
    "lrp_transforms",
    "gaussian_conditional",
)

# Top-level rename map applied AFTER per-slice cc_/lrp_/gaussian_conditional
# rerooting. Keys are matched as exact prefixes (with the trailing dot).
_UPSTREAM_TOP_LEVEL_RENAMES: Dict[str, str] = {
    "h_a.": "latent_codec.h_a.",
    "h_mean_s.": "latent_codec.h_s.h_mean_s.",
    "h_scale_s.": "latent_codec.h_s.h_scale_s.",
    "entropy_bottleneck.": "latent_codec.z.entropy_bottleneck.",
}

# Upstream STF places the WindowAttention parameters directly under
# ``conv_b.<i>.attn.{qkv,proj,relative_position_*}``. CompressAI wraps the
# WindowAttention inside a :class:`compressai.layers.attn.swin.WMSA` shim, so
# the live model keeps ``WMSA.attn = WindowAttention(...)`` and the
# parameters land at ``conv_b.<i>.attn.attn.*``. This regex inserts the extra
# ``.attn`` so renamed upstream keys round-trip into the WMSA wrapper without
# changing the model topology.
_WMSA_NEST_PATTERN = re.compile(
    r"(\.conv_b\.\d+\.attn)\.(qkv\.|proj\.|relative_position_)"
)


def _nest_winmsa_keys(key: str) -> str:
    """Insert the WMSA wrapper level (``.attn``) into upstream
    ``conv_b.*.attn.{qkv,proj,relative_position_*}`` keys."""
    return _WMSA_NEST_PATTERN.sub(r"\1.attn.\2", key)


def convert_upstream_stf_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate a candidate ``STF`` / ``WACNN`` state dict into compressai layout.

    Upstream checkpoints (``stf_<bpp>_best.pth.tar`` / ``cnn_<bpp>_best.pth.tar``
    from `Zou et al. 2022 <https://arxiv.org/abs/2203.08450>`_) are saved from a
    ``DataParallel``-wrapped module and place the channel-conditional entropy
    transforms at the model root. After the H+G containerised refactor
    compressai houses those transforms (plus the Gaussian conditional and
    the hyperprior backbone) inside ``latent_codec.*``. This helper:

    - strips the leading ``module.`` prefix added by ``DataParallel``;
    - re-roots ``cc_mean_transforms.{k}`` / ``cc_scale_transforms.{k}`` /
      ``lrp_transforms.{k}`` under
      ``latent_codec.y.channel_context.y{k}.{mean_cc,scale_cc}.*`` /
      ``latent_codec.y.latent_codec.y{k}.lrp_transform.*``;
    - replicates the single shared ``gaussian_conditional.*`` buffer set
      under each per-slice leaf (``latent_codec.y.latent_codec.y{k}.gaussian_conditional.*``);
    - moves ``entropy_bottleneck.*`` / ``h_a.*`` / ``h_mean_s.*`` /
      ``h_scale_s.*`` under ``latent_codec.*`` per the new layout;
    - leaves ``g_a`` / ``g_s`` / ``patch_embed`` / ``layers`` /
      ``syn_layers`` / ``end_conv`` keys unchanged.

    .. caveat::
       The Phase 3 wiring sets ``emit_mean_support=True`` on the
       ``MeanScaleContextHead`` so the upstream LRP layout
       (``cat(latent_means, *prev_y_hat, y_hat)``) is recoverable inside the
       leaf — upstream ``lrp_transforms.{k}`` weights therefore transfer
       byte-for-byte. The model's ``WinNoShiftAttention`` consumers wrap
       their windowed-attention layers in a :class:`WMSA` shim, so the
       conversion also nests upstream ``conv_b.{i}.attn.{qkv,proj,
       relative_position_*}`` keys under the extra ``.attn`` level (see
       :func:`_nest_winmsa_keys`).

    The returned dict can be loaded by :meth:`WACNN.from_state_dict` or
    :meth:`SymmetricalTransFormer.from_state_dict`. Both ``from_state_dict``
    entry points auto-detect the upstream layout and call this helper, so
    direct invocation is only needed when persisting the converted dict.
    """
    converted: Dict[str, Tensor] = {}

    _LEGACY_ROOT_HEADS = set(_UPSTREAM_LATENT_CODEC_PREFIXES) | {
        "h_a",
        "h_mean_s",
        "h_scale_s",
        "entropy_bottleneck",
    }

    # Pass 1: strip module. prefix, fold the upstream single-``attn`` window
    # attention path back into compressai's WMSA wrapper layout, and inventory
    # which keys exist.
    cleaned: Dict[str, Tensor] = {}
    has_legacy_root_keys = False
    for key, value in state_dict.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        new_key = _nest_winmsa_keys(new_key)
        cleaned[new_key] = value
        if new_key.split(".", 1)[0] in _LEGACY_ROOT_HEADS:
            has_legacy_root_keys = True

    if not has_legacy_root_keys:
        # Already in (or near) the new layout — return cleaned dict as-is.
        return cleaned

    # Pass 2: discover slice indices to drive gaussian_conditional replication
    # and per-slice rerooting.
    slice_indices = sorted(
        {
            int(key.split(".")[1])
            for key in cleaned
            if key.startswith("cc_mean_transforms.")
        }
    )
    num_slices = len(slice_indices)

    for key, value in cleaned.items():
        head = key.split(".", 1)[0]
        if head == "cc_mean_transforms":
            _, k, *rest = key.split(".")
            new_key = f"latent_codec.y.channel_context.y{k}.mean_cc." + ".".join(rest)
            converted[new_key] = value
        elif head == "cc_scale_transforms":
            _, k, *rest = key.split(".")
            new_key = f"latent_codec.y.channel_context.y{k}.scale_cc." + ".".join(rest)
            converted[new_key] = value
        elif head == "lrp_transforms":
            _, k, *rest = key.split(".")
            new_key = f"latent_codec.y.latent_codec.y{k}.lrp_transform." + ".".join(
                rest
            )
            converted[new_key] = value
        elif head == "gaussian_conditional":
            # Replicate the single shared instance to per-slice leaves.
            tail = key[len("gaussian_conditional.") :]
            for k in range(num_slices):
                new_key = (
                    f"latent_codec.y.latent_codec.y{k}" f".gaussian_conditional.{tail}"
                )
                converted[new_key] = value
        else:
            renamed = key
            for prefix, replacement in _UPSTREAM_TOP_LEVEL_RENAMES.items():
                if key.startswith(prefix):
                    renamed = replacement + key[len(prefix) :]
                    break
            converted[renamed] = value

    return converted


def _is_upstream_stf_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic: upstream checkpoints either carry a ``module.`` prefix or
    place ``cc_mean_transforms`` at the root instead of under ``latent_codec``.
    """
    for key in state_dict:
        if key.startswith("module."):
            return True
        head = key.split(".", 1)[0]
        if head in _UPSTREAM_LATENT_CODEC_PREFIXES or head in {
            "h_a",
            "h_mean_s",
            "h_scale_s",
            "entropy_bottleneck",
        }:
            return True
    return False


@register_model("stf-wacnn")
class WACNN(CompressionModel):
    r"""WACNN model from R. Zou, C. Song, Z. Zhang: `"The Devil Is in the
    Details: Window-based Attention for Image Compression"
    <https://arxiv.org/abs/2203.08450>`_, IEEE/CVF Conf. on Computer Vision
    and Pattern Recognition (CVPR), 2022.

    CNN-based variant that inserts window-based attention modules
    (:class:`compressai.layers.attn.WinNoShiftAttention` with
    ``output_proj=False``) inside the analysis/synthesis transforms, paired
    with a Minnen2020-style channel-wise autoregressive entropy model.

    The entropy stack is a fully containerised
    :class:`HyperpriorLatentCodec` that owns ``h_a``, ``h_s``, the ``z``
    bottleneck and the per-slice ``ChannelGroupsLatentCodec`` running in
    Family 1 ``side_in_context=True`` mode.

    Args:
        N (int): Number of channels in the hyperprior backbone.
        M (int): Number of channels in the latent representation.
        num_slices (int): Number of channel slices for the entropy model.
    """

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        num_slices: int = 10,
        max_support_slices: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")
        slice_ch = M // num_slices

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            WinNoShiftAttention(
                dim=N, num_heads=8, window_size=8, shift_size=4, output_proj=False
            ),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            WinNoShiftAttention(
                dim=M, num_heads=8, window_size=4, shift_size=2, output_proj=False
            ),
        )
        self.g_s = nn.Sequential(
            WinNoShiftAttention(
                dim=M, num_heads=8, window_size=4, shift_size=2, output_proj=False
            ),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            WinNoShiftAttention(
                dim=N, num_heads=8, window_size=8, shift_size=4, output_proj=False
            ),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        h_a = nn.Sequential(
            conv3x3(M, M),
            nn.GELU(),
            conv3x3(M, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )
        h_mean_s = _build_stf_h_subpel(N, M)
        h_scale_s = _build_stf_h_subpel(N, M)

        self.latent_codec = _build_family1_latent_codec(
            N=N,
            M=M,
            slice_ch=slice_ch,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            widths=(224, 176, 128, 64),
            h_a=h_a,
            h_mean_s=h_mean_s,
            h_scale_s=h_scale_s,
        )

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
        shape: Dict[str, Tuple[int, ...]] | Tuple[int, int],
    ) -> Dict[str, Tensor]:
        y_out = self.latent_codec.decompress(strings, shape)
        return {"x_hat": self.g_s(y_out["y_hat"]).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "WACNN":
        if _is_upstream_stf_state_dict(state_dict):
            state_dict = convert_upstream_stf_state_dict(state_dict)
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.7.weight"].size(0)
        num_slices = infer_num_slices(state_dict) or 10
        max_support_slices = infer_max_support_slices(state_dict, M, num_slices)
        net = cls(
            N=N,
            M=M,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
        )
        net.load_state_dict(state_dict)
        return net


def _build_stf_h_subpel(N: int, M: int) -> nn.Sequential:
    """Default ``h_mean_s`` / ``h_scale_s`` stack used by both WACNN and
    SymmetricalTransFormer's WACNN-shaped variant: 5 conv / subpel blocks
    going from ``N -> N -> 224 -> 256 -> 288 -> M`` with GELU activations.
    """
    return nn.Sequential(
        conv3x3(N, N),
        nn.GELU(),
        subpel_conv3x3(N, 224, 2),
        nn.GELU(),
        conv3x3(224, 256),
        nn.GELU(),
        subpel_conv3x3(256, 288, 2),
        nn.GELU(),
        conv3x3(288, M),
    )


def _build_stf_transformer_h_subpel(
    bottleneck_channels: int, latent_channels: int, embed_dim: int
) -> nn.Sequential:
    """Hyper-synthesis stack used by :class:`SymmetricalTransFormer`.

    Mirrors the original Zou et al. STF Transformer configuration: widths
    derived from the per-stage channel counts (``latent_channels - k *
    embed_dim``) instead of the WACNN-style fixed ladder.
    """
    return nn.Sequential(
        conv3x3(bottleneck_channels, latent_channels - 3 * embed_dim),
        nn.GELU(),
        subpel_conv3x3(
            latent_channels - 3 * embed_dim, latent_channels - 2 * embed_dim, 2
        ),
        nn.GELU(),
        conv3x3(latent_channels - 2 * embed_dim, latent_channels - embed_dim),
        nn.GELU(),
        subpel_conv3x3(latent_channels - embed_dim, latent_channels, 2),
        nn.GELU(),
        conv3x3(latent_channels, latent_channels),
    )


def _build_family1_latent_codec(
    *,
    N: int,
    M: int,
    slice_ch: int,
    num_slices: int,
    max_support_slices: int,
    widths: Sequence[int],
    h_a: nn.Module,
    h_mean_s: nn.Module,
    h_scale_s: nn.Module,
) -> HyperpriorLatentCodec:
    """Assemble the Phase 3 Family 1 entropy stack: ``HyperpriorLatentCodec``
    wrapping ``DualHyperSynthesis`` and a per-slice
    ``ChannelGroupsLatentCodec`` (``side_in_context=True``) whose channel
    contexts are :class:`MeanScaleContextHead` instances and leaves are
    :class:`LRPGaussianLatentCodec` (STE-quantised). ``side_channels = 2 *
    M`` because ``DualHyperSynthesis`` cats ``h_mean_s(z_hat)`` and
    ``h_scale_s(z_hat)``.

    The channel-context heads run with ``emit_mean_support=True`` so each
    head appends ``cat(latent_means, *prev_y_hat)`` to its output; the leaf
    splits that trailing block off (``mean_support_trail_channels``) and
    uses it as the LRP input. This reproduces the upstream STF / WACNN LRP
    layout (``cat(latent_means, *prev_y_hat, y_hat)``), so the
    ``lrp_transforms.{k}`` weights from upstream Zou et al. checkpoints
    transfer byte-for-byte after the rename pass in
    :func:`convert_upstream_stf_state_dict`.
    """
    side_channels = 2 * M

    def _support_count(k: int) -> int:
        if max_support_slices < 0:
            return k
        return min(k, max_support_slices)

    def _mean_support_ch(k: int) -> int:
        # cat(latent_means(M), *prev_y_hat(slice_ch * support_count)).
        return M + slice_ch * _support_count(k)

    def _leaf(k: int, _slice_ch: int) -> LRPGaussianLatentCodec:
        ms_ch = _mean_support_ch(k)
        return LRPGaussianLatentCodec(
            lrp_transform=make_entropy_transform(
                ms_ch + _slice_ch,  # cat(mean_support, y_hat)
                _slice_ch,
                widths=widths,
            ),
            mean_support_trail_channels=ms_ch,
            quantizer="ste",
        )

    def _channel_context(_k: int, _slice_ch: int, support_ch: int) -> nn.Module:
        return build_mean_scale_head(
            slice_ch=_slice_ch,
            support_ch=support_ch,
            widths=widths,
            side_split=M,
            emit_mean_support=True,
        )

    return HyperpriorLatentCodec(
        h_a=h_a,
        h_s=DualHyperSynthesis(h_mean_s, h_scale_s),
        latent_codec={
            "z": EntropyBottleneckLatentCodec(
                entropy_bottleneck=EntropyBottleneck(N), quantizer="noise"
            ),
            "y": build_channel_slice_codec(
                groups=[slice_ch] * num_slices,
                side_channels=side_channels,
                side_in_context=True,
                max_support_slices=max_support_slices,
                leaf_factory=_leaf,
                channel_context_factory=_channel_context,
            ),
        },
    )


@register_model("stf")
class SymmetricalTransFormer(CompressionModel):
    r"""Symmetrical Transformer model (STF) from R. Zou, C. Song, Z. Zhang:
    `"The Devil Is in the Details: Window-based Attention for Image
    Compression" <https://arxiv.org/abs/2203.08450>`_, IEEE/CVF Conf. on
    Computer Vision and Pattern Recognition (CVPR), 2022.

    Transformer-based companion of :class:`WACNN` that builds the
    analysis/synthesis transforms with stacked Swin-style basic layers and a
    channel-wise autoregressive entropy model. The entropy stack mirrors
    :class:`WACNN`'s containerised :class:`HyperpriorLatentCodec` (Family 1
    ``side_in_context=True`` mode), with widths derived from the
    transformer's stage channel counts.

    Args:
        embed_dim (int): Patch-embedding dimension.
        num_slices (int): Number of channel slices for the entropy model.
    """

    def __init__(
        self,
        pretrain_img_size: int = 256,
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 48,
        depths: Optional[Sequence[int]] = None,
        num_heads: Optional[Sequence[int]] = None,
        window_size: int = 4,
        num_slices: int = 12,
        max_support_slices: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        patch_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        depths = list(depths or [2, 2, 6, 2])
        num_heads = list(num_heads or [3, 6, 12, 24])
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.patch_embed = _PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for layer_index in range(self.num_layers):
            self.layers.append(
                _STFBasicLayer(
                    dim=int(embed_dim * 2**layer_index),
                    depth=depths[layer_index],
                    num_heads=num_heads[layer_index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:layer_index]) : sum(depths[: layer_index + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=None
                    if layer_index == self.num_layers - 1
                    else PatchMerging,
                )
            )

        reversed_depths = list(reversed(depths))
        reversed_heads = list(reversed(num_heads))
        self.syn_layers = nn.ModuleList()
        for layer_index in range(self.num_layers):
            self.syn_layers.append(
                _STFBasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - layer_index)),
                    depth=reversed_depths[layer_index],
                    num_heads=reversed_heads[layer_index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(reversed_depths[:layer_index]) : sum(
                            reversed_depths[: layer_index + 1]
                        )
                    ],
                    norm_layer=norm_layer,
                    downsample=None
                    if layer_index == self.num_layers - 1
                    else PatchSplit,
                )
            )

        self.end_conv = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim * patch_size**2, kernel_size=5, stride=1, padding=2
            ),
            nn.PixelShuffle(patch_size),
            nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
        )

        latent_channels = int(embed_dim * 2 ** (self.num_layers - 1))
        bottleneck_channels = latent_channels // 2
        if latent_channels % num_slices != 0:
            raise ValueError("latent_channels must be divisible by num_slices")
        slice_ch = latent_channels // num_slices
        resolved_max_support = (
            num_slices // 2 if max_support_slices is None else max_support_slices
        )

        h_a = nn.Sequential(
            conv3x3(latent_channels, latent_channels),
            nn.GELU(),
            conv3x3(latent_channels, latent_channels - embed_dim),
            nn.GELU(),
            conv3x3(
                latent_channels - embed_dim, latent_channels - 2 * embed_dim, stride=2
            ),
            nn.GELU(),
            conv3x3(latent_channels - 2 * embed_dim, latent_channels - 3 * embed_dim),
            nn.GELU(),
            conv3x3(latent_channels - 3 * embed_dim, bottleneck_channels, stride=2),
        )
        h_mean_s = _build_stf_transformer_h_subpel(
            bottleneck_channels, latent_channels, embed_dim
        )
        h_scale_s = _build_stf_transformer_h_subpel(
            bottleneck_channels, latent_channels, embed_dim
        )

        self.latent_codec = _build_family1_latent_codec(
            N=bottleneck_channels,
            M=latent_channels,
            slice_ch=slice_ch,
            num_slices=num_slices,
            max_support_slices=resolved_max_support,
            widths=(224, 176, 128, 64),
            h_a=h_a,
            h_mean_s=h_mean_s,
            h_scale_s=h_scale_s,
        )

    def _analysis_transform(self, x: Tensor) -> Tuple[Tensor, int, int]:
        output = self.patch_embed(x)
        height, width = output.size(2), output.size(3)
        output = self.pos_drop(output.flatten(2).transpose(1, 2))
        for layer in self.layers:
            output, height, width = layer(output, height, width)
        channels = self.embed_dim * 2 ** (self.num_layers - 1)
        output = (
            output.view(-1, height, width, channels).permute(0, 3, 1, 2).contiguous()
        )
        return output, height, width

    def _synthesis_transform(self, y_hat: Tensor, height: int, width: int) -> Tensor:
        channels = self.embed_dim * 2 ** (self.num_layers - 1)
        output = (
            y_hat.permute(0, 2, 3, 1).contiguous().view(-1, height * width, channels)
        )
        for layer in self.syn_layers:
            output, height, width = layer(output, height, width)
        output = (
            output.view(-1, height, width, self.embed_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return self.end_conv(output)

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y, height, width = self._analysis_transform(x)
        y_out = self.latent_codec(y)
        return {
            "x_hat": self._synthesis_transform(y_out["y_hat"], height, width),
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y, _, _ = self._analysis_transform(x)
        y_out = self.latent_codec.compress(y)
        return {"strings": y_out["strings"], "shape": y_out["shape"]}

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Dict[str, Tuple[int, ...]] | Tuple[int, int],
    ) -> Dict[str, Tensor]:
        y_out = self.latent_codec.decompress(strings, shape)
        y_hat = y_out["y_hat"]
        height, width = y_hat.shape[2:]
        return {"x_hat": self._synthesis_transform(y_hat, height, width).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "SymmetricalTransFormer":
        if _is_upstream_stf_state_dict(state_dict):
            state_dict = convert_upstream_stf_state_dict(state_dict)
        patch_size = state_dict["patch_embed.proj.weight"].size(2)
        embed_dim = state_dict["patch_embed.proj.weight"].size(0)
        layer_indices = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith("layers.") and ".blocks." in key
            }
        )
        depths = [
            len(
                {
                    int(key.split(".")[3])
                    for key in state_dict
                    if key.startswith(f"layers.{layer_index}.blocks.")
                }
            )
            for layer_index in layer_indices
        ]
        num_heads = [
            state_dict[
                f"layers.{layer_index}.blocks.0.attn.relative_position_bias_table"
            ].size(1)
            for layer_index in layer_indices
        ]
        table_size = state_dict[
            "layers.0.blocks.0.attn.relative_position_bias_table"
        ].size(0)
        window_size = (math.isqrt(table_size) + 1) // 2
        num_slices = infer_num_slices(state_dict) or 12
        latent_channels = embed_dim * 2 ** (len(depths) - 1)
        max_support_slices = infer_max_support_slices(
            state_dict, latent_channels, num_slices
        )

        net = cls(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
        )
        net.load_state_dict(state_dict)
        return net
