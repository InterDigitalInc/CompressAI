"""DCAE (Dictionary-based Channel-wise Auto-regressive Entropy) model.

J. Lu, L. Zhang, X. Zhou, M. Li, W. Li, S. Gu, "Learned Image Compression
with Dictionary-based Entropy Model", IEEE/CVF Conf. on Computer Vision
and Pattern Recognition (CVPR), 2025
(https://arxiv.org/abs/2504.00496).

Adapted from the upstream reference implementation; the entropy stack uses
the containerized
:class:`~compressai.latent_codecs.HyperpriorLatentCodec` /
:class:`~compressai.latent_codecs.ChannelGroupsLatentCodec` wiring shared
with STF / WACNN / TCM / CCA, plus the dictionary cross-attention head
:class:`~compressai.models._helpers.dictionary_context.DictionaryMeanScaleContextHead`
introduced for DCAE / SAAF.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import DropPath
from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    EntropyBottleneckLatentCodec,
    GaussianConditionalLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers.attn.dictionary import ConvolutionalGLU, Scale
from compressai.layers.attn.swin import pad_to_window_multiple
from compressai.models._helpers.dictionary_context import (
    SharedDictionary,
    build_dictionary_mean_scale_head,
)
from compressai.models._helpers.slice_helpers import (
    infer_num_slices,
    lrp_support_channels,
    make_entropy_transform,
)
from compressai.models.base import CompressionModel
from compressai.models.sensetime import ResidualBottleneckBlock
from compressai.models.utils import conv, deconv
from compressai.registry import register_model

__all__ = ["DCAE"]


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
        support = [y_hat_[i] for i in self.support_slices[k]]
        if not support:
            return self.channel_context[f"y{k}"](side_params)
        return self.channel_context[f"y{k}"](
            self.merge_params(side_params, self.merge_y(*support))
        )


# ---------------------------------------------------------------------------
# DCAE-private g_a / g_s building blocks
# (Inlined from the upstream DCAE source rather than lifted to compressai/layers/
# because they are not reused by other models in the PR series.)
# ---------------------------------------------------------------------------


class _ResidualBottleneckBlockWithStride(nn.Module):
    """DCAE stride-2 residual-bottleneck downsampling block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = conv(in_ch, out_ch, kernel_size=5, stride=2)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.conv(input_tensor)
        output = self.res1(output)
        output = self.res2(output)
        return self.res3(output)


class _ResidualBottleneckBlockWithUpsample(nn.Module):
    """DCAE residual-bottleneck upsampling block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = deconv(in_ch, out_ch, kernel_size=5, stride=2)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.res1(input_tensor)
        output = self.res2(output)
        output = self.res3(output)
        return self.conv(output)


class _WMSA(nn.Module):
    """Windowed multi-head self-attention with optional cyclic shift.

    Lifted verbatim from the upstream DCAE source. ``type`` is ``"W"`` for a
    plain window-attention pass or ``"SW"`` for a shifted-window pass.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        type: str,
    ) -> None:
        super().__init__()
        if type not in {"W", "SW"}:
            raise ValueError(f"Unsupported attention type: {type}")
        if input_dim % head_dim != 0:
            raise ValueError("input_dim must be divisible by head_dim")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        relative_position = torch.zeros(
            self.n_heads, 2 * window_size - 1, 2 * window_size - 1
        )
        nn.init.trunc_normal_(relative_position, std=0.02)
        self.relative_position_params = nn.Parameter(relative_position)
        self.linear = nn.Linear(input_dim, output_dim)

    def generate_mask(
        self,
        height_windows: int,
        width_windows: int,
        window_size: int,
        shift: int,
    ) -> Tensor:
        attention_mask = torch.zeros(
            height_windows,
            width_windows,
            window_size,
            window_size,
            window_size,
            window_size,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attention_mask

        split = window_size - shift
        attention_mask[-1, :, :split, :, split:, :] = True
        attention_mask[-1, :, split:, :, :split, :] = True
        attention_mask[:, -1, :, :split, :, split:] = True
        attention_mask[:, -1, :, split:, :, :split] = True
        return rearrange(attention_mask, "h w p1 p2 p3 p4 -> 1 1 (h w) (p1 p2) (p3 p4)")

    def relative_embedding(self) -> Tensor:
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(
                    self.window_size, device=self.relative_position_params.device
                ),
                torch.arange(
                    self.window_size, device=self.relative_position_params.device
                ),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        relation = coords[:, None, :] - coords[None, :, :] + self.window_size - 1
        return self.relative_position_params[
            :, relation[:, :, 0].long(), relation[:, :, 1].long()
        ]

    def forward(self, input_tensor: Tensor) -> Tensor:
        if self.type != "W":
            input_tensor = torch.roll(
                input_tensor,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )

        output = rearrange(
            input_tensor,
            "b (h p1) (w p2) c -> b h w p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        height_windows = output.size(1)
        width_windows = output.size(2)
        output = rearrange(
            output,
            "b h w p1 p2 c -> b (h w) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )

        qkv = self.embedding_layer(output)
        qkv = rearrange(
            qkv,
            "b nw np (three heads dim) -> three b heads nw np dim",
            three=3,
            heads=self.n_heads,
            dim=self.head_dim,
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        similarity = torch.einsum("bhwnc,bhwmc->bhwnm", query, key) * self.scale
        similarity = similarity + rearrange(
            self.relative_embedding(), "h p q -> 1 h 1 p q"
        )
        if self.type != "W":
            attention_mask = self.generate_mask(
                height_windows,
                width_windows,
                self.window_size,
                shift=self.window_size // 2,
            )
            similarity = similarity.masked_fill(attention_mask, float("-inf"))

        probabilities = similarity.softmax(dim=-1)
        output = torch.einsum("bhwij,bhwjc->bhwic", probabilities, value)
        output = rearrange(output, "b h w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (h w) (p1 p2) c -> b (h p1) (w p2) c",
            h=height_windows,
            p1=self.window_size,
            p2=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output


class _ResScaleConvolutionGateBlock(nn.Module):
    """Residual-scaled WMSA + ConvolutionalGLU MLP block (channel-last)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        type: str = "W",
        input_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        del output_dim, input_resolution
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = _WMSA(input_dim, input_dim, head_dim, window_size, type)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = ConvolutionalGLU(input_dim, input_dim * 4)
        self.res_scale_1 = Scale(input_dim, init_value=1.0)
        self.res_scale_2 = Scale(input_dim, init_value=1.0)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.res_scale_1(input_tensor) + self.drop_path(
            self.msa(self.ln1(input_tensor))
        )
        return self.res_scale_2(output) + self.drop_path(self.mlp(self.ln2(output)))


class _SwinBlockWithConvMulti(nn.Module):
    """Stack of ``block_num`` WMSA layers (W / SW alternating) followed by a 3x3 conv."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        block: type[nn.Module] = _ResScaleConvolutionGateBlock,
        block_num: int = 2,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        self.layers = nn.ModuleList(
            block(
                input_dim,
                input_dim,
                head_dim,
                window_size,
                drop_path,
                type="W" if index % 2 == 0 else "SW",
            )
            for index in range(block_num)
        )
        self.block_num = block_num
        self.conv = conv(input_dim, output_dim, 3, 1)
        self.window_size = window_size

    def forward(self, input_tensor: Tensor) -> Tensor:
        output, pad_height, pad_width = pad_to_window_multiple(
            input_tensor, self.window_size
        )
        output = rearrange(output, "b c h w -> b h w c")
        for layer in self.layers:
            output = layer(output)
        output = rearrange(output, "b h w c -> b c h w")
        output = self.conv(output) + F.pad(input_tensor, (0, pad_width, 0, pad_height))
        if pad_height > 0 or pad_width > 0:
            output = output[:, :, : input_tensor.size(2), : input_tensor.size(3)]
        return output.contiguous()


# ---------------------------------------------------------------------------
# DCAE model
# ---------------------------------------------------------------------------


@register_model("dcae")
class DCAE(CompressionModel):
    """DCAE model (Lu et al., CVPR 2025).

    Containerized entropy stack:

    .. code-block:: text

        latent_codec = HyperpriorLatentCodec(
            h_a=h_a,
            h_s=_DualHyperSynthesis(h_mean_s, h_scale_s),  # original h_z_s2/h_z_s1 swapped to means/scales
            latent_codec={
                "z": EntropyBottleneckLatentCodec(EntropyBottleneck(N), quantizer="noise"),
                "y": _SideContextChannelGroupsLatentCodec(...) wired
                     inline with per-slice DictionaryMeanScaleContextHead +
                     _LRPGaussianLatentCodec(mean_support_trail_channels=...).
            },
        )

    The shared dictionary tensor ``dt`` lives at ``self.shared_dictionary.dt``
    (a single state-dict path); each per-slice
    :class:`DictionaryMeanScaleContextHead` accesses it via a closure to
    avoid duplicating the parameter under K paths.
    """

    def __init__(
        self,
        head_dim: Optional[Sequence[int]] = None,
        N: int = 192,
        M: int = 320,
        hyper_channels: int = 192,
        num_slices: int = 5,
        max_support_slices: int = 5,
        feature_dims: Optional[Sequence[int]] = None,
        block_num: Optional[Sequence[int]] = None,
        dict_num: int = 128,
        dict_head_num: int = 20,
        dictionary_dim: Optional[int] = None,
        window_size: int = 8,
        hyper_window_size: int = 4,
        hyper_head_dim: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        head_dim = tuple(head_dim or (8, 16, 32, 32, 16, 8))
        feature_dims = tuple(feature_dims or (96, 144, 256))
        block_num = tuple(block_num or (1, 2, 12))
        dictionary_dim = dictionary_dim or 32 * dict_head_num
        if len(head_dim) != 6:
            raise ValueError("head_dim must have six entries")
        if len(feature_dims) != 3:
            raise ValueError("feature_dims must have three entries")
        if len(block_num) != 3:
            raise ValueError("block_num must have three entries")
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")

        self.N = int(N)
        self.M = int(M)
        self.hyper_channels = int(hyper_channels)
        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)
        self.head_dim = head_dim
        self.feature_dims = feature_dims
        self.block_num = block_num
        self.dict_num = int(dict_num)
        self.dict_head_num = int(dict_head_num)
        self.dictionary_dim = int(dictionary_dim)
        self.window_size = int(window_size)
        self.hyper_window_size = int(hyper_window_size)
        self.hyper_head_dim = int(hyper_head_dim)

        slice_channels = M // num_slices
        input_image_channel = 3
        output_image_channel = 3

        # ----- g_a / g_s -----
        self.g_a = nn.Sequential(
            _ResidualBottleneckBlockWithStride(input_image_channel, feature_dims[0]),
            _SwinBlockWithConvMulti(
                feature_dims[0],
                feature_dims[0],
                head_dim[0],
                self.window_size,
                0.0,
                _ResScaleConvolutionGateBlock,
                block_num=block_num[0],
            ),
            _ResidualBottleneckBlockWithStride(feature_dims[0], feature_dims[1]),
            _SwinBlockWithConvMulti(
                feature_dims[1],
                feature_dims[1],
                head_dim[1],
                self.window_size,
                0.0,
                _ResScaleConvolutionGateBlock,
                block_num=block_num[1],
            ),
            _ResidualBottleneckBlockWithStride(feature_dims[1], feature_dims[2]),
            _SwinBlockWithConvMulti(
                feature_dims[2],
                feature_dims[2],
                head_dim[2],
                self.window_size,
                0.0,
                _ResScaleConvolutionGateBlock,
                block_num=block_num[2],
            ),
            conv(feature_dims[2], M, kernel_size=5, stride=2),
        )
        self.g_s = nn.Sequential(
            deconv(M, feature_dims[2], kernel_size=5, stride=2),
            _SwinBlockWithConvMulti(
                feature_dims[2],
                feature_dims[2],
                head_dim[3],
                self.window_size,
                0.0,
                _ResScaleConvolutionGateBlock,
                block_num=block_num[2],
            ),
            _ResidualBottleneckBlockWithUpsample(feature_dims[2], feature_dims[1]),
            _SwinBlockWithConvMulti(
                feature_dims[1],
                feature_dims[1],
                head_dim[4],
                self.window_size,
                0.0,
                _ResScaleConvolutionGateBlock,
                block_num=block_num[1],
            ),
            _ResidualBottleneckBlockWithUpsample(feature_dims[1], feature_dims[0]),
            _SwinBlockWithConvMulti(
                feature_dims[0],
                feature_dims[0],
                head_dim[5],
                self.window_size,
                0.0,
                _ResScaleConvolutionGateBlock,
                block_num=block_num[0],
            ),
            _ResidualBottleneckBlockWithUpsample(feature_dims[0], output_image_channel),
        )

        # ----- h_a / h_mean_s / h_scale_s -----
        h_a = nn.Sequential(
            _ResidualBottleneckBlockWithStride(M, N),
            _SwinBlockWithConvMulti(
                N,
                N,
                hyper_head_dim,
                hyper_window_size,
                0.0,
                _ResScaleConvolutionGateBlock,
                block_num=1,
            ),
            conv(N, hyper_channels, kernel_size=3, stride=2),
        )

        # NOTE: upstream DCAE used h_z_s1 for *scales* and h_z_s2 for *means*;
        # _DualHyperSynthesis(h_mean_s, h_scale_s) flips this to the
        # (means, scales) ordering shared with STF / TCM / CCA. The convert
        # script renames h_z_s2 -> h_s.h_mean_s and h_z_s1 -> h_s.h_scale_s.
        h_mean_s = nn.Sequential(
            deconv(hyper_channels, N, kernel_size=3, stride=2),
            _SwinBlockWithConvMulti(
                N,
                N,
                hyper_head_dim,
                hyper_window_size,
                0.0,
                _ResScaleConvolutionGateBlock,
                block_num=1,
            ),
            _ResidualBottleneckBlockWithUpsample(N, M),
        )
        h_scale_s = nn.Sequential(
            deconv(hyper_channels, N, kernel_size=3, stride=2),
            _SwinBlockWithConvMulti(
                N,
                N,
                hyper_head_dim,
                hyper_window_size,
                0.0,
                _ResScaleConvolutionGateBlock,
                block_num=1,
            ),
            _ResidualBottleneckBlockWithUpsample(N, M),
        )

        # ----- Shared dictionary -----
        self.shared_dictionary = SharedDictionary(
            dict_num=self.dict_num, dictionary_dim=self.dictionary_dim
        )

        # ----- Latent codec -----
        cross_attention_kwargs = {
            "head_num": self.dict_head_num,
            "mlp_rate": 4,
            "qkv_bias": True,
            "dictionary_dim": self.dictionary_dim,
        }

        widths = (224, 128)
        groups = [slice_channels] * num_slices

        def support_count(k: int) -> int:
            return k if max_support_slices < 0 else min(k, max_support_slices)

        # mean_support_trail = support tensor produced by the head:
        # cat(input(2M + slice_ch*support_count), dict_info(M)).
        def mean_support_ch(k: int) -> int:
            return 2 * M + slice_channels * support_count(k) + M

        support_slices = [list(range(support_count(k))) for k in range(num_slices)]

        # Side-parameter channel-groups wiring, inlined ELIC-style (mirrors
        # STF / TCM). channel_context covers y0..y(K-1); each head sees
        # cat(side_params(2M), *prev_y_hat) and emits cat(scale, mean,
        # mean_support) for the LRP-aware leaf. The dictionary cross-attention
        # head (DCAE's distinctive piece) replaces STF's plain mean/scale head.
        channel_context = {
            f"y{k}": build_dictionary_mean_scale_head(
                slice_ch=slice_channels,
                support_ch=2 * M + slice_channels * support_count(k),
                shared_dictionary=self.shared_dictionary,
                dict_output_ch=M,
                cross_attention_kwargs=cross_attention_kwargs,
                widths=widths,
                emit_mean_support=True,
            )
            for k in range(num_slices)
        }
        y_latent_codec = {
            f"y{k}": _LRPGaussianLatentCodec(
                lrp_transform=make_entropy_transform(
                    lrp_support_channels(2 * M, slice_channels, k, max_support_slices)
                    + M,
                    slice_channels,
                    widths=widths,
                ),
                lrp_scale=0.5,
                mean_support_trail_channels=mean_support_ch(k),
                chunks=("scales", "means"),
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
                    quantizer="noise",
                ),
                "y": _SideContextChannelGroupsLatentCodec(
                    groups=groups,
                    channel_context=channel_context,
                    latent_codec=y_latent_codec,
                    support_slices=support_slices,
                ),
            },
        )

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        y = self.g_a(x)
        out = self.latent_codec(y)
        return {
            "x_hat": self.g_s(out["y_hat"]),
            "likelihoods": out["likelihoods"],
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        return self.latent_codec.compress(y)

    def decompress(
        self, strings: Sequence[Sequence[bytes]], shape: Sequence[int]
    ) -> Dict[str, Tensor]:
        out = self.latent_codec.decompress(strings, shape)
        return {"x_hat": self.g_s(out["y_hat"]).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "DCAE":
        kwargs = _infer_config_from_state_dict(state_dict)
        net = cls(**kwargs)
        net.load_state_dict(state_dict)
        return net


# ---------------------------------------------------------------------------
# Architecture inference helpers (state_dict -> hyperparameters)
# ---------------------------------------------------------------------------


def _infer_stage_block_num(state_dict: Dict[str, Tensor], prefix: str) -> int:
    """Count ``{prefix}{N}.ln1.weight`` entries to recover SwinBlockWithConvMulti depth."""
    matches = [
        k for k in state_dict if k.startswith(prefix) and k.endswith(".ln1.weight")
    ]
    return len(matches)


def _infer_attention_head_dim(
    state_dict: Dict[str, Tensor], prefix: str, channel_count: int
) -> int:
    """Recover head_dim from ``{prefix}.layers.0.msa.relative_position_params``."""
    key = f"{prefix}.layers.0.msa.relative_position_params"
    if key not in state_dict:
        raise KeyError(f"missing {key} for head-dim inference")
    n_heads = state_dict[key].size(0)
    if channel_count % n_heads != 0:
        raise ValueError(
            f"channel_count {channel_count} not divisible by n_heads {n_heads} at {prefix}"
        )
    return channel_count // n_heads


def _infer_window_size(state_dict: Dict[str, Tensor], prefix: str) -> int:
    """Recover window_size from ``{prefix}.layers.0.msa.relative_position_params`` shape."""
    key = f"{prefix}.layers.0.msa.relative_position_params"
    if key not in state_dict:
        raise KeyError(f"missing {key} for window-size inference")
    relative_dim = state_dict[key].size(1)
    if relative_dim % 2 == 0:
        raise ValueError(
            f"relative_position_params has even spatial dim {relative_dim} at {prefix}"
        )
    return (relative_dim + 1) // 2


def _infer_config_from_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, object]:
    """Recover DCAE constructor kwargs from a containerized state_dict."""
    feature_dims = (
        state_dict["g_a.0.conv.weight"].size(0),
        state_dict["g_a.2.conv.weight"].size(0),
        state_dict["g_a.4.conv.weight"].size(0),
    )
    N = state_dict["latent_codec.h_a.0.conv.weight"].size(0)
    M = state_dict["latent_codec.h_a.0.conv.weight"].size(1)
    hyper_channels = state_dict["latent_codec.z.entropy_bottleneck.quantiles"].size(0)
    num_slices = infer_num_slices(state_dict)
    slice_ch = M // num_slices
    # Recover max_support_slices from the widest cc input width.
    # support_ch (head input) = 2 * M + slice_ch * support_count.
    # cc_in_ch (mean_cc.0.weight) = support_ch + M = 3 * M + slice_ch * support_count.
    if num_slices > 1:
        widest = max(
            state_dict[f"latent_codec.y.channel_context.y{k}.mean_cc.0.weight"].size(1)
            for k in range(num_slices)
        )
        support_count = (widest - 3 * M) // slice_ch
        max_support_slices = max(support_count, 1)
    else:
        max_support_slices = 1

    block_num = (
        _infer_stage_block_num(state_dict, "g_a.1.layers."),
        _infer_stage_block_num(state_dict, "g_a.3.layers."),
        _infer_stage_block_num(state_dict, "g_a.5.layers."),
    )
    head_dim = (
        _infer_attention_head_dim(state_dict, "g_a.1", feature_dims[0]),
        _infer_attention_head_dim(state_dict, "g_a.3", feature_dims[1]),
        _infer_attention_head_dim(state_dict, "g_a.5", feature_dims[2]),
        _infer_attention_head_dim(state_dict, "g_s.1", feature_dims[2]),
        _infer_attention_head_dim(state_dict, "g_s.3", feature_dims[1]),
        _infer_attention_head_dim(state_dict, "g_s.5", feature_dims[0]),
    )

    dt = state_dict["shared_dictionary.dt"]
    dict_num = dt.size(0)
    dictionary_dim = dt.size(1)
    dict_head_num = state_dict[
        "latent_codec.y.channel_context.y0.cross_attention.scale"
    ].size(0)

    return dict(
        head_dim=head_dim,
        N=N,
        M=M,
        hyper_channels=hyper_channels,
        num_slices=num_slices,
        max_support_slices=max_support_slices,
        feature_dims=feature_dims,
        block_num=block_num,
        dict_num=dict_num,
        dict_head_num=dict_head_num,
        dictionary_dim=dictionary_dim,
        window_size=_infer_window_size(state_dict, "g_a.1"),
        hyper_window_size=_infer_window_size(state_dict, "latent_codec.h_a.1"),
        hyper_head_dim=_infer_attention_head_dim(state_dict, "latent_codec.h_a.1", N),
    )
