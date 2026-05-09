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

import re

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

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
from compressai.layers import (
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.layers.attn import ConvTransBlock, SWAtten
from compressai.models._helpers.channel_context import build_mean_scale_head
from compressai.models._helpers.channel_slice import build_channel_slice_codec
from compressai.models.base import CompressionModel
from compressai.models.utils import conv
from compressai.registry import register_model

__all__ = [
    "TCM",
    "convert_upstream_tcm_state_dict",
]


# ----------------------------------------------------------------------------
# Upstream LIC_TCM checkpoint conversion
# ----------------------------------------------------------------------------


# Heads from upstream LIC_TCM (Liu et al. 2023) checkpoints that move under
# ``latent_codec.*`` after the H+G containerised refactor.
_UPSTREAM_LATENT_CODEC_PREFIXES = (
    "cc_mean_transforms",
    "cc_scale_transforms",
    "lrp_transforms",
    "atten_mean",
    "atten_scale",
    "mean_support_transforms",
    "scale_support_transforms",
    "gaussian_conditional",
)

# Top-level rename map applied AFTER per-slice rerooting. Keys are matched as
# exact prefixes (with the trailing dot).
_UPSTREAM_TOP_LEVEL_RENAMES: Dict[str, str] = {
    "h_a.": "latent_codec.h_a.",
    "h_mean_s.": "latent_codec.h_s.h_mean_s.",
    "h_scale_s.": "latent_codec.h_s.h_scale_s.",
    "entropy_bottleneck.": "latent_codec.z.entropy_bottleneck.",
}

# Upstream LIC_TCM wraps each ``SWAtten`` in an ``nn.Sequential`` and stores
# parameters at ``atten_mean.{k}.0.<...>``. Compressai's :class:`SWAtten`
# lives directly at ``mean_support_transform.<...>`` after rerooting, so the
# leading ``.0`` wrapper level is stripped.
_UPSTREAM_SWATTEN_WRAPPER = re.compile(
    r"^(atten_mean|atten_scale|mean_support_transforms|scale_support_transforms)\.(\d+)\.0\."
)


def _rename_msa_keys(key: str, value: Tensor) -> Tuple[str, Tensor]:
    """Translate upstream LIC_TCM ConvTransBlock-internal MSA layout to
    compressai's :class:`WMSA` wrapper layout.

    Three kinds of upstream keys appear inside ``g_a`` / ``g_s`` / ``h_a`` /
    ``h_mean_s`` / ``h_scale_s`` blocks:

    - ``.msa.relative_position_params`` is a ``(2*win-1, 2*win-1, num_heads)``
      buffer; compressai's ``WindowAttention`` registers it as a flat
      ``(N, num_heads)`` ``relative_position_bias_table``. The value is
      permuted and reshaped accordingly.
    - ``.msa.embedding_layer`` is upstream's name for the fused ``qkv``
      linear; compressai exposes it as ``.msa.attn.qkv.<...>``.
    - ``.msa.linear`` is upstream's optional output projection; compressai
      drops it and instead uses the WindowAttention's identity ``.proj`` —
      see :func:`_ensure_identity_attention_projection` for the identity
      injection that keeps strict ``load_state_dict`` round-trips clean.
    """
    if ".msa.relative_position_params" in key:
        new_key = key.replace(
            ".msa.relative_position_params",
            ".msa.attn.relative_position_bias_table",
        )
        new_value = value.permute(1, 2, 0).reshape(-1, value.size(0)).contiguous()
        return new_key, new_value
    if ".msa.embedding_layer." in key:
        return key.replace(".msa.embedding_layer.", ".msa.attn.qkv."), value
    if ".msa.linear." in key:
        return key.replace(".msa.linear.", ".msa.output_proj."), value
    return key, value


def _ensure_identity_attention_projection(
    state_dict: Dict[str, Tensor],
    output_proj_key: str,
    output_proj_value: Tensor,
) -> None:
    """Inject an identity ``WindowAttention.proj`` for upstream blocks whose
    output projection sits outside the attention module (``.msa.linear`` →
    ``.msa.output_proj``). The model has both ``.msa.attn.proj`` (inside
    WindowAttention, identity-initialised here) and ``.msa.output_proj``
    (the actual learned projection) so strict ``load_state_dict`` succeeds.
    """
    prefix, suffix = output_proj_key.rsplit(".msa.output_proj.", 1)
    attn_proj_key = f"{prefix}.msa.attn.proj.{suffix}"
    if attn_proj_key in state_dict:
        return
    if suffix == "weight":
        dimension = output_proj_value.size(0)
        state_dict[attn_proj_key] = torch.eye(
            dimension,
            dtype=output_proj_value.dtype,
            device=output_proj_value.device,
        )
        return
    if suffix == "bias":
        state_dict[attn_proj_key] = torch.zeros_like(output_proj_value)


def _is_upstream_tcm_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic: upstream LIC_TCM checkpoints either carry the ``module.``
    prefix from ``DataParallel`` saving, the ``.msa.relative_position_params``
    buffer, or the per-slice entropy heads (``cc_mean_transforms`` /
    ``atten_mean`` / ``lrp_transforms`` / ``gaussian_conditional`` / ``h_a``
    / ``h_mean_s`` / ``h_scale_s`` / ``entropy_bottleneck``) at the model
    root rather than under ``latent_codec.*``.
    """
    legacy_roots = set(_UPSTREAM_LATENT_CODEC_PREFIXES) | {
        "h_a",
        "h_mean_s",
        "h_scale_s",
        "entropy_bottleneck",
    }
    for key in state_dict:
        if key.startswith("module."):
            return True
        if (
            ".msa.relative_position_params" in key
            or ".msa.embedding_layer." in key
            or ".msa.linear." in key
        ):
            return True
        if key.split(".", 1)[0] in legacy_roots:
            return True
    return False


def convert_upstream_tcm_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream LIC_TCM state dict into compressai layout.

    Upstream checkpoints (e.g. ``0.013.pth..tar`` from
    `Liu et al. 2023 <https://arxiv.org/abs/2303.14978>`_,
    https://github.com/jmliu206/LIC_TCM) place the channel-conditional entropy
    transforms and the hyperprior backbone at the model root. After the H+G
    containerised refactor compressai houses those transforms (plus the
    Gaussian conditional and the ``z`` bottleneck) inside ``latent_codec.*``.
    This helper:

    - strips the leading ``module.`` prefix added by ``DataParallel``;
    - rewrites ConvTransBlock attention buffers via :func:`_rename_msa_keys`
      (``.msa.relative_position_params`` /  ``.msa.embedding_layer`` /
      ``.msa.linear``) and standard layer-name renames (``ln1`` → ``norm1``,
      ``mlp.0`` / ``mlp.2`` → ``mlp.fc1`` / ``mlp.fc2``);
    - unwraps the upstream ``nn.Sequential`` wrapper around each ``SWAtten``
      (``atten_mean.{k}.0.<...>`` → ``atten_mean.{k}.<...>``);
    - re-roots ``cc_mean_transforms.{k}`` / ``cc_scale_transforms.{k}`` /
      ``lrp_transforms.{k}`` under
      ``latent_codec.y.channel_context.y{k}.{mean_cc,scale_cc}.*`` /
      ``latent_codec.y.latent_codec.y{k}.lrp_transform.*``;
    - re-roots ``atten_mean.{k}`` / ``atten_scale.{k}`` (or their
      ``mean_support_transforms`` / ``scale_support_transforms`` aliases)
      under ``latent_codec.y.channel_context.y{k}.{mean,scale}_support_transform.*``;
    - replicates the single shared ``gaussian_conditional.*`` buffer set
      under each per-slice leaf
      (``latent_codec.y.latent_codec.y{k}.gaussian_conditional.*``);
    - moves ``entropy_bottleneck.*`` / ``h_a.*`` / ``h_mean_s.*`` /
      ``h_scale_s.*`` under ``latent_codec.*`` per the new layout;
    - leaves ``g_a`` / ``g_s`` keys (other than the MSA renames inside their
      ConvTransBlocks) untouched.

    The Phase 3 wiring sets ``emit_mean_support=True`` on the
    :class:`MeanScaleContextHead`, so the upstream LRP layout
    (``cat(latent_means, *prev_y_hat, y_hat)``) is recoverable inside the
    leaf — upstream ``lrp_transforms.{k}`` weights therefore transfer
    byte-for-byte.

    The returned dict can be loaded by :meth:`TCM.from_state_dict`, which
    auto-detects the upstream layout and calls this helper, so direct
    invocation is only needed when persisting the converted dict.
    """
    # Pass 1: strip ``module.`` prefix; rewrite ConvTransBlock attention
    # buffers and layer names; unwrap the SWAtten ``nn.Sequential`` wrapper;
    # alias ``atten_mean`` / ``atten_scale`` to the canonical
    # ``mean_support_transforms`` / ``scale_support_transforms`` names so the
    # per-slice rerooting in Pass 2 only has to handle one form.
    cleaned: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        new_key, value = _rename_msa_keys(new_key, value)
        wrapper = _UPSTREAM_SWATTEN_WRAPPER.match(new_key)
        if wrapper:
            new_key = (
                f"{wrapper.group(1)}.{wrapper.group(2)}." + new_key[wrapper.end() :]
            )
        if new_key.startswith("atten_mean."):
            new_key = "mean_support_transforms." + new_key[len("atten_mean.") :]
        elif new_key.startswith("atten_scale."):
            new_key = "scale_support_transforms." + new_key[len("atten_scale.") :]
        new_key = new_key.replace(".ln1.", ".norm1.")
        new_key = new_key.replace(".ln2.", ".norm2.")
        new_key = new_key.replace(".mlp.0.", ".mlp.fc1.")
        new_key = new_key.replace(".mlp.2.", ".mlp.fc2.")
        if ".msa.output_proj." in new_key:
            _ensure_identity_attention_projection(cleaned, new_key, value)
        cleaned[new_key] = value

    # Pass 2: discover slice indices to drive ``gaussian_conditional``
    # replication, then reroot per-slice and top-level keys.
    converted: Dict[str, Tensor] = {}
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
        elif head == "mean_support_transforms":
            _, k, *rest = key.split(".")
            new_key = (
                f"latent_codec.y.channel_context.y{k}.mean_support_transform."
                + ".".join(rest)
            )
            converted[new_key] = value
        elif head == "scale_support_transforms":
            _, k, *rest = key.split(".")
            new_key = (
                f"latent_codec.y.channel_context.y{k}.scale_support_transform."
                + ".".join(rest)
            )
            converted[new_key] = value
        elif head == "lrp_transforms":
            _, k, *rest = key.split(".")
            new_key = f"latent_codec.y.latent_codec.y{k}.lrp_transform." + ".".join(
                rest
            )
            converted[new_key] = value
        elif head == "gaussian_conditional":
            tail = key[len("gaussian_conditional.") :]
            for k in range(num_slices):
                new_key = (
                    f"latent_codec.y.latent_codec.y{k}.gaussian_conditional.{tail}"
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
class TCM(CompressionModel):
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
    bottleneck and the per-slice ``ChannelGroupsLatentCodec`` running in
    Family 1 ``side_in_context=True`` mode. The channel-context heads run
    with ``support_transform_factory=SWAtten`` so per-slice ``mean_in`` /
    ``scale_in`` are routed through independent SWAtten instances before the
    3-conv ``mean_cc`` / ``scale_cc`` stacks (TCM's distinctive widths
    ``(224, 128)``).

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
        self.latent_codec = _build_tcm_latent_codec(
            hyper_channels=hyper_channels,
            M=M,
            slice_ch=slice_ch,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            window_size=self.window_size,
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
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "TCM":
        if _is_upstream_tcm_state_dict(state_dict):
            state_dict = convert_upstream_tcm_state_dict(state_dict)
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


def _build_tcm_latent_codec(
    *,
    hyper_channels: int,
    M: int,
    slice_ch: int,
    num_slices: int,
    max_support_slices: int,
    window_size: int,
    h_a: nn.Module,
    h_mean_s: nn.Module,
    h_scale_s: nn.Module,
) -> HyperpriorLatentCodec:
    """Assemble TCM's Family 1 entropy stack: ``HyperpriorLatentCodec``
    wrapping ``DualHyperSynthesis`` and a per-slice
    ``ChannelGroupsLatentCodec`` (``side_in_context=True``).

    Differences from the STF / WACNN wiring (see
    :func:`compressai.models.stf._build_family1_latent_codec`):

    - ``widths=(224, 128)`` — TCM's 3-conv ``mean_cc`` / ``scale_cc`` stack
      in place of STF's 5-conv ``(224, 176, 128, 64)`` ladder.
    - ``support_transform_factory=SWAtten`` — independent windowed-attention
      transforms wrap each slice's ``mean_in`` / ``scale_in`` before the
      conv stack. The two SWAtten instances per slice mirror upstream
      ``atten_mean[k]`` / ``atten_scale[k]``.

    Like STF, the channel-context heads run with ``emit_mean_support=True``
    and the leaves with matching ``mean_support_trail_channels`` so the
    upstream LRP layout (``cat(latent_means, *prev_y_hat, y_hat)``) is
    preserved — upstream ``lrp_transforms.{k}`` weights transfer
    byte-for-byte after :func:`convert_upstream_tcm_state_dict`.
    """
    widths = (224, 128)
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

    def _swatten_factory(c_in: int, c_out: int) -> nn.Module:
        return SWAtten(
            input_dim=c_in,
            output_dim=c_out,
            head_dim=16,
            window_size=window_size,
            drop_path=0.0,
            inter_dim=128,
        )

    def _channel_context(_k: int, _slice_ch: int, support_ch: int) -> nn.Module:
        return build_mean_scale_head(
            slice_ch=_slice_ch,
            support_ch=support_ch,
            widths=widths,
            side_split=M,
            emit_mean_support=True,
            support_transform_factory=_swatten_factory,
        )

    return HyperpriorLatentCodec(
        h_a=h_a,
        h_s=DualHyperSynthesis(h_mean_s, h_scale_s),
        latent_codec={
            "z": EntropyBottleneckLatentCodec(
                entropy_bottleneck=EntropyBottleneck(hyper_channels),
                quantizer="noise",
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
