"""SAAF (Sparse Attention with Adaptive Frequency) model.

H. Ma, X. Shi, H. Sun, X. Yue, X. Liu, G. Wang, W. Cai, "Learned Image
Compression via Sparse Attention and Adaptive Frequency", IEEE/CVF Conf.
on Computer Vision and Pattern Recognition (CVPR), 2026.

Adapted from the upstream reference implementation at
https://github.com/huidong-ma/SAAF
based on CompressAI, DCAE, and AuxT.

SAAF combines adaptive-frequency auxiliary transform branches
(``aux_enc`` / ``aux_dec``) with a denoising regularizer that produces
``diffusion_loss`` during training.
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
from compressai.models._helpers.auxt import OLP
from compressai.models._helpers.auxt import aux_loss as _aggregate_aux_loss
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

__all__ = ["SAAF"]


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
# SAAF-private g_a / g_s building blocks
# ---------------------------------------------------------------------------


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Largest divisor of ``channels`` not exceeding ``max_groups``.

    Used to size GroupNorm groups inside :class:`_DenoisingAsRegularizer`.
    """
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBottleneckBlockWithStride(nn.Module):
    """SAAF stride-2 residual-bottleneck downsampling block."""

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
    """SAAF residual-bottleneck upsampling block."""

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


class _AdaptiveFrequencyBlock(nn.Module):
    """SAAF analysis-side AuxT block: frequency-attention mixer + OLP.

    Used inside ``aux_enc`` to produce an auxiliary feature stream that is
    summed into ``g_a`` at every stage boundary. Holds an
    :class:`OLP` whose orthogonality regulariser is collected by
    :meth:`SAAF.aux_loss`.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.olp = OLP(in_dim, out_dim)
        mid_dim = max(in_dim // 4, 4)
        self.freq_attn = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, 4, 1),
            nn.Softmax(dim=1),
        )
        self.freq_weights = nn.Parameter(torch.tensor([1.0, 0.8, 0.8, 0.6]))

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.shape
        frequency_attention = self.freq_attn(input_tensor)
        frequency_weights = torch.exp(self.freq_weights).view(1, 4, 1, 1)
        output = input_tensor.unsqueeze(1) * frequency_attention.unsqueeze(2)
        output = output * frequency_weights.unsqueeze(2)
        output = output.sum(dim=1)
        output = output.flatten(2).permute(0, 2, 1)
        output = self.olp(output)
        return output.permute(0, 2, 1).view(batch_size, -1, height, width)


class _InverseAdaptiveFrequencyBlock(nn.Module):
    """Synthesis-side counterpart of :class:`_AdaptiveFrequencyBlock`.

    Used inside ``aux_dec``. Adds a small frequency-attention residual on
    top of the OLP output (parameterised by ``0.1``) so the synthesis
    branch can reweight subbands without dominating the main ``g_s`` path.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.olp = OLP(in_dim, out_dim)
        mid_dim = max(in_dim // 4, 4)
        self.freq_attn = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, 4, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.shape
        frequency_weights = self.freq_attn(input_tensor)
        output = input_tensor.flatten(2).permute(0, 2, 1)
        output = self.olp(output)
        output = output.permute(0, 2, 1).view(batch_size, -1, height, width)
        enhanced = output * frequency_weights.mean(dim=1, keepdim=True)
        return output + 0.1 * enhanced


class _DenoisingAsRegularizer(nn.Module):
    """Noise-prediction head producing SAAF's training-only ``diffusion_loss``.

    Conditions on ``z_hat`` (hyperprior latent), perturbs the encoder
    latent ``y`` with random Gaussian noise scaled by a per-batch random
    timestep, and asks a small UNet-style predictor to recover the noise.
    The MSE of the prediction is returned as a scalar regulariser. Lifted
    verbatim from the upstream SAAF reference implementation.
    """

    def __init__(self, latent_dim: int = 320, hyper_channels: int = 192) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.noise_predictor = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(_group_count(latent_dim), latent_dim),
            nn.SiLU(),
            ResidualBottleneckBlock(latent_dim, latent_dim),
            ResidualBottleneckBlock(latent_dim, latent_dim),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(_group_count(latent_dim), latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1),
        )
        condition_channels = max(latent_dim * 4 // 5, 4)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(hyper_channels, condition_channels, 1),
            nn.GroupNorm(_group_count(condition_channels), condition_channels),
            nn.GELU(),
            nn.Conv2d(condition_channels, latent_dim, 3, padding=1),
            nn.Dropout(0.1),
            nn.GELU(),
        )

    def forward(self, latent: Tensor, hyper_latent: Tensor) -> Tensor:
        batch_size, channels, height, width = latent.size()
        condition = self.condition_encoder(hyper_latent)
        condition = F.interpolate(
            condition, size=(height, width), mode="bilinear", align_corners=False
        )
        time = torch.rand(batch_size, 1, device=latent.device, dtype=latent.dtype)
        noise = torch.randn_like(latent)
        noisy_latent = latent + noise * time.view(batch_size, 1, 1, 1)
        time_embedding = self.time_embed(time).view(batch_size, channels, 1, 1)
        prediction = self.noise_predictor(noisy_latent + time_embedding + condition)
        return F.mse_loss(prediction, noise)


class _CrossSparseWindowAttention(nn.Module):
    """SAAF-specific windowed attention with shared global tokens.

    Differs from :class:`compressai.layers.attn.swin.WMSA` in two ways:
    (1) each window mixes its local self-attention output with a global
    attention pass against a small set of learned tokens (parameterised
    by ``num_global_tokens`` and a learnable ``global_alpha``); (2) uses
    a flat ``relative_position_bias_table`` indexed by a precomputed
    ``relative_position_index``, matching the upstream layout (so SAAF
    checkpoints round-trip without further key renames).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        num_global_tokens: int = 2,
    ) -> None:
        super().__init__()
        if input_dim % head_dim != 0:
            raise ValueError("input_dim must be divisible by head_dim")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, input_dim))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)
        self.global_kv = nn.Linear(input_dim, input_dim * 2, bias=False)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.linear = nn.Linear(input_dim, output_dim)
        self.register_buffer("global_alpha", torch.tensor(0.25))

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, height, width, channels = input_tensor.shape
        window_size = self.window_size
        height_windows = height // window_size
        width_windows = width // window_size
        output = input_tensor.view(
            batch_size,
            height_windows,
            window_size,
            width_windows,
            window_size,
            channels,
        )
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
        output = output.view(
            batch_size * height_windows * width_windows,
            window_size * window_size,
            channels,
        )
        num_windows = height_windows * width_windows

        qkv = self.embedding_layer(output).reshape(
            batch_size * num_windows,
            window_size * window_size,
            3,
            self.n_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2]

        similarity = torch.einsum("bhpc,bhqc->bhpq", query * self.scale, key)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(window_size * window_size, window_size * window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        similarity = similarity + relative_position_bias.unsqueeze(0)
        probabilities = similarity.softmax(dim=-1)
        output_local = torch.einsum("bhij,bhjc->bhic", probabilities, value)

        global_tokens = self.global_tokens.expand(batch_size * num_windows, -1, -1)
        global_tokens = global_tokens + output.mean(dim=1, keepdim=True)
        global_kv = self.global_kv(global_tokens).reshape(
            batch_size * num_windows,
            self.num_global_tokens,
            2,
            self.n_heads,
            self.head_dim,
        )
        global_kv = global_kv.permute(2, 0, 3, 1, 4).contiguous()
        key_global, value_global = global_kv[0], global_kv[1]
        similarity_global = torch.einsum(
            "bhpc,bhgc->bhpg", query * self.scale, key_global
        )
        probabilities_global = similarity_global.softmax(dim=-1)
        output_global = torch.einsum(
            "bhpg,bhgc->bhpc", probabilities_global, value_global
        )

        output = (
            1 - self.global_alpha
        ) * output_local + self.global_alpha * output_global
        output = output.transpose(1, 2).reshape(
            batch_size * num_windows, window_size * window_size, channels
        )
        output = self.linear(output)
        output = output.view(
            batch_size,
            height_windows,
            width_windows,
            window_size,
            window_size,
            channels,
        )
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
        return output.view(batch_size, height, width, channels)


class _SpatialAttentionLayer(nn.Module):
    """SAAF-specific transformer layer: ``_CrossSparseWindowAttention`` +
    :class:`compressai.layers.attn.dictionary.ConvolutionalGLU` MLP block.
    Counterpart of DCAE's ``_ResScaleConvolutionGateBlock``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        input_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        del output_dim, input_resolution
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = _CrossSparseWindowAttention(
            input_dim, input_dim, head_dim, window_size
        )
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


class _SpatialAttentionBlock(nn.Module):
    """Stack of ``block_num`` :class:`_SpatialAttentionLayer` instances followed
    by a 3x3 conv. Counterpart of DCAE's ``_SwinBlockWithConvMulti``."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        block: type[nn.Module] = _SpatialAttentionLayer,
        block_num: int = 2,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        self.layers = nn.ModuleList(
            block(input_dim, input_dim, head_dim, window_size, drop_path)
            for _ in range(block_num)
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
# SAAF model
# ---------------------------------------------------------------------------


@register_model("saaf")
class SAAF(CompressionModel):
    """SAAF model (Ma et al., CVPR 2026).

    Containerized entropy stack identical to DCAE (see
    :class:`compressai.models.dcae.DCAE`); SAAF differs only in the
    ``g_a`` / ``g_s`` building blocks, the parallel ``aux_enc`` /
    ``aux_dec`` AuxT chain (each block carrying an :class:`OLP`), and the
    training-only :class:`_DenoisingAsRegularizer` ``diffusion_prior`` head.
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

        # ----- g_a / g_s (SAAF-specific spatial-attention stages) -----
        self.m_down1 = [
            _SpatialAttentionBlock(
                feature_dims[0],
                feature_dims[0],
                head_dim[0],
                self.window_size,
                0.0,
                block=_SpatialAttentionLayer,
                block_num=block_num[0],
            ),
            _ResidualBottleneckBlockWithStride(feature_dims[0], feature_dims[1]),
        ]
        self.m_down2 = [
            _SpatialAttentionBlock(
                feature_dims[1],
                feature_dims[1],
                head_dim[1],
                self.window_size,
                0.0,
                block=_SpatialAttentionLayer,
                block_num=block_num[1],
            ),
            _ResidualBottleneckBlockWithStride(feature_dims[1], feature_dims[2]),
        ]
        self.m_down3 = [
            _SpatialAttentionBlock(
                feature_dims[2],
                feature_dims[2],
                head_dim[2],
                self.window_size,
                0.0,
                block=_SpatialAttentionLayer,
                block_num=block_num[2],
            ),
            conv(feature_dims[2], M, kernel_size=5, stride=2),
        ]
        self.g_a = nn.Sequential(
            _ResidualBottleneckBlockWithStride(input_image_channel, feature_dims[0]),
            *self.m_down1,
            *self.m_down2,
            *self.m_down3,
        )
        self.aux_enc = nn.ModuleList(
            [
                _AdaptiveFrequencyBlock(input_image_channel, feature_dims[0]),
                _AdaptiveFrequencyBlock(feature_dims[0], feature_dims[1]),
                _AdaptiveFrequencyBlock(feature_dims[1], feature_dims[2]),
                _AdaptiveFrequencyBlock(feature_dims[2], M),
            ]
        )

        self.m_up1 = [
            _SpatialAttentionBlock(
                feature_dims[2],
                feature_dims[2],
                head_dim[3],
                self.window_size,
                0.0,
                block=_SpatialAttentionLayer,
                block_num=block_num[2],
            ),
            _ResidualBottleneckBlockWithUpsample(feature_dims[2], feature_dims[1]),
        ]
        self.m_up2 = [
            _SpatialAttentionBlock(
                feature_dims[1],
                feature_dims[1],
                head_dim[4],
                self.window_size,
                0.0,
                block=_SpatialAttentionLayer,
                block_num=block_num[1],
            ),
            _ResidualBottleneckBlockWithUpsample(feature_dims[1], feature_dims[0]),
        ]
        self.m_up3 = [
            _SpatialAttentionBlock(
                feature_dims[0],
                feature_dims[0],
                head_dim[5],
                self.window_size,
                0.0,
                block=_SpatialAttentionLayer,
                block_num=block_num[0],
            ),
            _ResidualBottleneckBlockWithUpsample(feature_dims[0], output_image_channel),
        ]
        self.g_s = nn.Sequential(
            deconv(M, feature_dims[2], kernel_size=5, stride=2),
            *self.m_up1,
            *self.m_up2,
            *self.m_up3,
        )
        self.aux_dec = nn.ModuleList(
            [
                _InverseAdaptiveFrequencyBlock(M, feature_dims[2]),
                _InverseAdaptiveFrequencyBlock(feature_dims[2], feature_dims[1]),
                _InverseAdaptiveFrequencyBlock(feature_dims[1], feature_dims[0]),
                _InverseAdaptiveFrequencyBlock(feature_dims[0], output_image_channel),
            ]
        )

        # ----- h_a / h_mean_s / h_scale_s (same SAAF blocks, hyper config) -----
        h_a = nn.Sequential(
            _ResidualBottleneckBlockWithStride(M, N),
            _SpatialAttentionBlock(
                N,
                N,
                hyper_head_dim,
                hyper_window_size,
                0.0,
                block=_SpatialAttentionLayer,
                block_num=1,
            ),
            conv(N, hyper_channels, kernel_size=3, stride=2),
        )

        # NOTE: upstream SAAF (like DCAE) uses h_z_s1 for *scales* and
        # h_z_s2 for *means*; _DualHyperSynthesis(h_mean_s, h_scale_s)
        # flips this. The convert script renames h_z_s2 -> h_s.h_mean_s
        # and h_z_s1 -> h_s.h_scale_s.
        h_mean_s = nn.Sequential(
            deconv(hyper_channels, N, kernel_size=3, stride=2),
            _SpatialAttentionBlock(
                N,
                N,
                hyper_head_dim,
                hyper_window_size,
                0.0,
                block=_SpatialAttentionLayer,
                block_num=1,
            ),
            _ResidualBottleneckBlockWithUpsample(N, M),
        )
        h_scale_s = nn.Sequential(
            deconv(hyper_channels, N, kernel_size=3, stride=2),
            _SpatialAttentionBlock(
                N,
                N,
                hyper_head_dim,
                hyper_window_size,
                0.0,
                block=_SpatialAttentionLayer,
                block_num=1,
            ),
            _ResidualBottleneckBlockWithUpsample(N, M),
        )

        # ----- Shared dictionary + diffusion prior -----
        self.shared_dictionary = SharedDictionary(
            dict_num=self.dict_num, dictionary_dim=self.dictionary_dim
        )
        self.diffusion_prior = _DenoisingAsRegularizer(
            latent_dim=M, hyper_channels=hyper_channels
        )

        # ----- Latent codec (identical to DCAE wiring) -----
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
        # STF / TCM / DCAE). channel_context covers y0..y(K-1); each head sees
        # cat(side_params(2M), *prev_y_hat) and emits cat(scale, mean,
        # mean_support) for the LRP-aware leaf. The dictionary cross-attention
        # head is the DCAE/SAAF-distinctive piece.
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

    @staticmethod
    def _merge_features(main: Tensor, auxiliary: Tensor) -> Tensor:
        """Sum ``auxiliary`` into ``main``, bilinear-interpolating to match
        spatial size when the AuxT branch hasn't downsampled yet."""
        if auxiliary.shape[2:] != main.shape[2:]:
            auxiliary = F.interpolate(
                auxiliary, size=main.shape[2:], mode="bilinear", align_corners=False
            )
        return main + auxiliary

    def _encode(self, x: Tensor) -> Tensor:
        """Main + AuxT analysis: walk ``g_a`` stage-by-stage, summing
        ``aux_enc[i]`` after every stage boundary."""
        y_main = self.g_a[0](x)
        y_aux = self.aux_enc[0](x)
        y_main = self._merge_features(y_main, y_aux)

        for index, stage in enumerate(
            (self.m_down1, self.m_down2, self.m_down3), start=1
        ):
            for layer in stage:
                y_main = layer(y_main)
            y_aux = self.aux_enc[index](y_aux)
            y_main = self._merge_features(y_main, y_aux)
        return y_main

    def _decode(self, y_hat: Tensor) -> Tensor:
        """Main + AuxT synthesis: mirror of :meth:`_encode`."""
        x_main = self.g_s[0](y_hat)
        x_aux = self.aux_dec[0](y_hat)
        x_main = self._merge_features(x_main, x_aux)

        for index, stage in enumerate((self.m_up1, self.m_up2, self.m_up3), start=1):
            for layer in stage:
                x_main = layer(x_main)
            x_aux = self.aux_dec[index](x_aux)
            x_main = self._merge_features(x_main, x_aux)
        return x_main

    def aux_loss(self) -> Tensor:
        """Auxiliary entropy-bottleneck loss plus OLP regulariser."""
        return super().aux_loss() + _aggregate_aux_loss(self)

    def forward(self, x: Tensor) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        y = self._encode(x)
        out = self.latent_codec(y)
        diffusion_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.training:
            # Reproduce upstream's z_hat-from-rounded-medians path so the
            # diffusion prior conditions on the same hyper latent the
            # entropy stack uses. Pulling z out of the latent codec keeps
            # the regulariser independent of the codec's noise/STE choice.
            z_hat = self.latent_codec.h_a(y)
            z_eb = self.latent_codec.latent_codec["z"].entropy_bottleneck
            z_medians = z_eb._get_medians()
            z_hat = torch.round(z_hat - z_medians) + z_medians
            diffusion_loss = self.diffusion_prior(y, z_hat)
        return {
            "x_hat": self._decode(out["y_hat"]),
            "likelihoods": out["likelihoods"],
            "diffusion_loss": diffusion_loss,
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self._encode(x)
        return self.latent_codec.compress(y)

    def decompress(
        self, strings: Sequence[Sequence[bytes]], shape: Sequence[int]
    ) -> Dict[str, Tensor]:
        out = self.latent_codec.decompress(strings, shape)
        return {"x_hat": self._decode(out["y_hat"]).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "SAAF":
        kwargs = _infer_config_from_state_dict(state_dict)
        net = cls(**kwargs)
        # ``_CrossSparseWindowAttention`` registers ``relative_position_index``
        # and ``global_alpha`` as non-persistent buffers, so they may be
        # absent from saved checkpoints. Tolerate the missing keys.
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            key
            for key in net.state_dict()
            if key.endswith("relative_position_index") or key.endswith("global_alpha")
        }
        missing_keys = set(incompatible_keys.missing_keys) - allowed_missing
        if missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                "Unexpected incompatibility while loading SAAF state_dict: "
                f"missing={sorted(missing_keys)}, "
                f"unexpected={sorted(incompatible_keys.unexpected_keys)}"
            )
        return net


# ---------------------------------------------------------------------------
# Architecture inference helpers (state_dict -> hyperparameters)
# ---------------------------------------------------------------------------


def _infer_stage_block_num(state_dict: Dict[str, Tensor], prefix: str) -> int:
    """Count ``{prefix}{N}.ln1.weight`` entries to recover SpatialAttentionBlock depth."""
    matches = [
        k for k in state_dict if k.startswith(prefix) and k.endswith(".ln1.weight")
    ]
    return len(matches)


def _infer_attention_head_dim(
    state_dict: Dict[str, Tensor], prefix: str, channel_count: int
) -> int:
    """Recover head_dim from ``{prefix}.layers.0.msa.relative_position_bias_table``."""
    key = f"{prefix}.layers.0.msa.relative_position_bias_table"
    if key not in state_dict:
        raise KeyError(f"missing {key} for head-dim inference")
    n_heads = state_dict[key].size(1)
    if channel_count % n_heads != 0:
        raise ValueError(
            f"channel_count {channel_count} not divisible by n_heads {n_heads} at {prefix}"
        )
    return channel_count // n_heads


def _infer_window_size(state_dict: Dict[str, Tensor], prefix: str) -> int:
    """Recover window_size from the
    ``relative_position_bias_table`` flat shape ``(2*win-1)^2``."""
    key = f"{prefix}.layers.0.msa.relative_position_bias_table"
    if key not in state_dict:
        raise KeyError(f"missing {key} for window-size inference")
    flat_dim = state_dict[key].size(0)
    side = int(round(flat_dim**0.5))
    if side * side != flat_dim or side % 2 == 0:
        raise ValueError(
            f"relative_position_bias_table has unexpected length {flat_dim} at {prefix}"
        )
    return (side + 1) // 2


def _infer_config_from_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, object]:
    """Recover SAAF constructor kwargs from a containerized state_dict."""
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
