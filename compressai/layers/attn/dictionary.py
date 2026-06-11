"""Dictionary-based multi-scale cross-attention building blocks.

These layers implement the entropy-side cross-attention used by the DCAE /
SAAF families, which factor a learned per-image dictionary
(``dt: nn.Parameter`` of shape ``(dict_num, dictionary_dim)``) shared across
slices and cross-attended by every channel-context head. They were lifted
from the upstream DCAE reference implementation (Lu et al., CVPR 2025); the
SAAF entropy stack (Ma et al., CVPR 2026) reuses the exact same blocks.

Adapted from the dictionary-entropy implementation released alongside the
DCAE / SAAF papers; transformer/attention plumbing follows their public
PyTorch sources.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor

__all__ = [
    "ConvWithDW",
    "ConvolutionalGLU",
    "DWConv",
    "DenseBlock",
    "MultiScaleAggregation",
    "MutiScaleDictionaryCrossAttentionGLU",
    "Scale",
    "SpatialAttentionModule",
]


class Scale(nn.Module):
    """Per-channel learnable scale (used as residual gating)."""

    def __init__(
        self, dim: int, init_value: float = 1.0, trainable: bool = True
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(dim),
            requires_grad=trainable,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * self.scale


class DWConv(nn.Module):
    """Depthwise 3x3 convolution operating on channel-last activations."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=dim,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = rearrange(input_tensor, "b h w c -> b c h w")
        output = self.dwconv(output)
        return rearrange(output, "b c h w -> b h w c")


class ConvolutionalGLU(nn.Module):
    """Convolutional Gated Linear Unit MLP block (channel-last)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = (hidden_features or in_features) // 2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output, gate = self.fc1(input_tensor).chunk(2, dim=-1)
        output = self.act(self.dwconv(output)) * gate
        return self.fc2(output)


class ConvWithDW(nn.Module):
    """1x1 -> depthwise 3x3 -> 1x1 conv block with GELU activations (channel-first)."""

    def __init__(self, input_dim: int = 320, output_dim: int = 320) -> None:
        super().__init__()
        self.in_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=True)
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(
            output_dim,
            output_dim,
            kernel_size=3,
            padding=1,
            groups=output_dim,
            bias=True,
        )
        self.act2 = nn.GELU()
        self.out_trans = nn.Conv2d(output_dim, output_dim, kernel_size=1, bias=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.in_trans(input_tensor)
        output = self.act1(output)
        output = self.dw_conv(output)
        output = self.act2(output)
        return self.out_trans(output)


class DenseBlock(nn.Module):
    """Dense block: ``layer_num`` ConvWithDW stages cat'd then projected back."""

    def __init__(self, dim: int = 320, layer_num: int = 3) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.conv_layers = nn.ModuleList(
            nn.Sequential(nn.GELU(), ConvWithDW(dim, dim)) for _ in range(layer_num)
        )
        self.proj = nn.Conv2d(dim * (layer_num + 1), dim, kernel_size=1, bias=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        outputs = [input_tensor]
        for layer in self.conv_layers:
            outputs.append(layer(outputs[-1]))
        return self.proj(torch.cat(outputs, dim=1))


class SpatialAttentionModule(nn.Module):
    """CBAM-style spatial attention map (avg + max pooled along channel axis)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: Tensor) -> Tensor:
        average = input_tensor.mean(dim=1, keepdim=True)
        maximum, _ = input_tensor.max(dim=1, keepdim=True)
        output = torch.cat([average, maximum], dim=1)
        return self.sigmoid(self.conv1(output))


class MultiScaleAggregation(nn.Module):
    """Combine 1x1 conv + DenseBlock with a CBAM spatial-attention gate."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.s = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = DenseBlock(dim)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = rearrange(input_tensor, "b h w c -> b c h w")
        output = self.s(output)
        output = self.dense(output)
        output = output * self.spatial_atte(output)
        return rearrange(output, "b c h w -> b h w c")


class MutiScaleDictionaryCrossAttentionGLU(nn.Module):
    """Cross-attend a per-slice support tensor against a shared dictionary.

    Used as the channel-context head body in DCAE / SAAF: ``input_tensor`` is
    the ``(B, input_dim, H, W)`` slice support and ``dictionary`` is the
    shared learnable ``(B, dict_num, dictionary_dim)`` dictionary tensor
    (typically materialised once per forward via
    ``dt.unsqueeze(0).expand(B, -1, -1)``). Returns ``(B, output_dim, H, W)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_rate: int = 4,
        head_num: int = 20,
        qkv_bias: bool = True,
        dictionary_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        dict_dim = dictionary_dim or 32 * head_num
        if dict_dim % head_num != 0:
            raise ValueError("dictionary_dim must be divisible by head_num")

        self.head_num = head_num
        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)
        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)
        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)
        self.mlp = ConvolutionalGLU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Sequential(nn.Linear(dict_dim, output_dim))
        self.softmax = nn.Softmax(dim=-1)
        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

    def forward(self, input_tensor: Tensor, dictionary: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.size()
        output = rearrange(input_tensor, "b c h w -> b h w c")
        output = self.x_trans(output)
        output = self.msa(self.ln_scale(output)) + self.res_scale_1(output)

        shortcut = output
        output = rearrange(self.q_trans(self.lnx(output)), "b h w c -> b c h w")
        query = rearrange(output, "b (e c) h w -> b e (h w) c", e=self.head_num)

        dictionary = self.dict_ln(dictionary)
        key = rearrange(self.k(dictionary), "b n (e c) -> b e n c", e=self.head_num)
        dictionary_value = rearrange(
            dictionary, "b n (e c) -> b e n c", e=self.head_num
        )

        scale = self.scale.to(device=query.device, dtype=query.dtype)
        similarity = torch.einsum("benc,bedc->bend", query, key) * scale
        probabilities = self.softmax(similarity)
        output = torch.einsum("bend,bedc->benc", probabilities, dictionary_value)
        output = rearrange(output, "b e (h w) c -> b h w (e c)", h=height, w=width)
        output = self.linear(output) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)
        output = self.output_trans(output)
        return rearrange(output, "b h w c -> b c h w", b=batch_size)
