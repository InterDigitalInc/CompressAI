from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, Mlp
from timm.models.swin_transformer import (
    WindowAttention as _TimmWindowAttention,
)
from timm.models.swin_transformer import (
    window_partition as _timm_window_partition,
)
from timm.models.swin_transformer import (
    window_reverse as _timm_window_reverse,
)
from torch import Tensor

from ..layers import AttentionBlock, ResidualBlock, conv1x1, conv3x3

__all__ = [
    "ConvTransBlock",
    "PatchMerging",
    "PatchSplit",
    "SWAtten",
    "SwinBlock",
    "WMSA",
    "WinNoShiftAttention",
    "WinResidualUnit",
    "WindowAttention",
    "build_window_attention_mask",
    "pad_to_window_multiple",
    "window_partition",
    "window_reverse",
]


def window_partition(input_tensor: Tensor, window_size: int) -> Tensor:
    """Square-window adapter around timm's ``window_partition``.

    timm uses ``Tuple[int, int]`` for the window size; the STF / WACNN models
    in compressai always use square windows, so this thin wrapper keeps the
    ``window_size: int`` call-site convention while delegating to timm.
    """
    return _timm_window_partition(input_tensor, (window_size, window_size))


def window_reverse(
    windows: Tensor,
    window_size: int,
    height: int,
    width: int,
) -> Tensor:
    """Square-window adapter around timm's ``window_reverse`` (see
    :func:`window_partition` for the rationale)."""
    return _timm_window_reverse(windows, (window_size, window_size), height, width)


def build_window_attention_mask(
    height: int,
    width: int,
    window_size: int,
    shift_size: int,
    device: torch.device,
) -> Optional[Tensor]:
    if shift_size == 0:
        return None

    img_mask = torch.zeros((1, height, width, 1), device=device)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )

    count = 0
    for h_index in h_slices:
        for w_index in w_slices:
            img_mask[:, h_index, w_index, :] = count
            count += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
    return attention_mask.masked_fill(attention_mask == 0, float(0.0))


def pad_to_window_multiple(
    input_tensor: Tensor,
    window_size: Union[int, Tuple[int, int]],
    *,
    layout: str = "BCHW",
) -> Tuple[Tensor, int, int]:
    """Right/bottom-pad a 4D tensor so its spatial dims are multiples of
    ``window_size``.

    Args:
        input_tensor: 4D tensor in either ``BCHW`` or ``BHWC`` layout.
        window_size: ``int`` (square window) or ``(window_h, window_w)``.
        layout: ``"BCHW"`` (default, PyTorch convention) or ``"BHWC"``
            (Swin / FTIC token-major layout).

    Returns:
        ``(padded_tensor, pad_h, pad_w)``, where ``pad_h`` / ``pad_w`` are
        the bottom / right padding widths added to the height / width
        dimension respectively.
    """
    if isinstance(window_size, int):
        win_h = win_w = int(window_size)
    else:
        win_h, win_w = (int(s) for s in window_size)

    if layout == "BCHW":
        height, width = input_tensor.shape[-2], input_tensor.shape[-1]
    elif layout == "BHWC":
        height, width = input_tensor.shape[1], input_tensor.shape[2]
    else:
        raise ValueError(f"layout must be 'BCHW' or 'BHWC', got {layout!r}")

    pad_h = (win_h - height % win_h) % win_h
    pad_w = (win_w - width % win_w) % win_w
    if pad_h == 0 and pad_w == 0:
        return input_tensor, 0, 0

    if layout == "BCHW":
        # F.pad on BCHW: (W_left, W_right, H_left, H_right)
        return F.pad(input_tensor, (0, pad_w, 0, pad_h)), pad_h, pad_w
    # F.pad on BHWC: (C_left, C_right, W_left, W_right, H_left, H_right)
    return F.pad(input_tensor, (0, 0, 0, pad_w, 0, pad_h)), pad_h, pad_w


class WindowAttention(_TimmWindowAttention):
    """timm ``WindowAttention`` with two minor tweaks for compressai:

    1. ``relative_position_index`` is re-registered as a *persistent* buffer
       so released compressai checkpoints (which include this tensor) load
       under ``strict=True``. timm registers it as ``persistent=False``.
    2. The constructor accepts an optional ``qk_scale`` to keep STF's
       (and CompressAI's) call-site convention; timm always derives the
       scale from ``head_dim``.

    Forward / state-dict layout otherwise match timm exactly, including
    the optional fused-attention path.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        if qk_scale is not None:
            self.scale = qk_scale
        # Promote the index buffer to persistent so checkpoint round-trip
        # works without filtering keys at load time.
        index = self.relative_position_index
        del self._buffers["relative_position_index"]
        self.register_buffer("relative_position_index", index, persistent=True)


class WMSA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int],
        head_dim: int,
        window_size: int,
        type: str = "W",
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        output_proj: bool = True,
    ) -> None:
        super().__init__()
        if type not in {"W", "SW"}:
            raise ValueError(f"Unsupported attention type: {type}")
        if input_dim % head_dim != 0:
            raise ValueError("`input_dim` must be divisible by `head_dim`.")

        self.window_size = window_size
        self.shift_size = 0 if type == "W" else window_size // 2
        self.attn = WindowAttention(
            dim=input_dim,
            window_size=window_size,
            num_heads=input_dim // head_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        # ``output_proj=False`` mirrors the STF / WACNN topology, which feeds
        # the WindowAttention output straight back into the downstream block
        # without an extra Linear projection. Set ``True`` (default) for the
        # SwinBlock / SWAtten variant used by the rest of CompressAI.
        self.output_proj = (
            nn.Linear(input_dim, output_dim or input_dim)
            if output_proj
            else nn.Identity()
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        _, height, width, _ = input_tensor.shape
        output, pad_height, pad_width = pad_to_window_multiple(
            input_tensor,
            self.window_size,
            layout="BHWC",
        )
        padded_height, padded_width = output.shape[1], output.shape[2]

        if self.shift_size > 0:
            mask = build_window_attention_mask(
                padded_height,
                padded_width,
                self.window_size,
                self.shift_size,
                output.device,
            )
            output = torch.roll(
                output,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            mask = None

        windows = window_partition(output, self.window_size)
        windows = windows.view(
            -1,
            self.window_size * self.window_size,
            windows.shape[-1],
        )
        windows = self.attn(windows, mask=mask)
        windows = windows.view(
            -1,
            self.window_size,
            self.window_size,
            windows.shape[-1],
        )
        output = window_reverse(windows, self.window_size, padded_height, padded_width)

        if self.shift_size > 0:
            output = torch.roll(
                output,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        if pad_height > 0 or pad_width > 0:
            output = output[:, :height, :width, :].contiguous()
        return self.output_proj(output)


class Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int],
        head_dim: int,
        window_size: int,
        drop_path: float,
        type: str = "W",
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.norm1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, type=type)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=int(input_dim * mlp_ratio),
            out_features=output_dim,
        )
        self.residual_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = input_tensor + self.drop_path(self.msa(self.norm1(input_tensor)))
        residual = self.residual_proj(output)
        return residual + self.drop_path(self.mlp(self.norm2(output)))


class SwinBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int],
        head_dim: int,
        window_size: int,
        drop_path: float,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.block_1 = Block(
            input_dim,
            input_dim,
            head_dim,
            window_size,
            drop_path,
            type="W",
            mlp_ratio=mlp_ratio,
        )
        self.block_2 = Block(
            input_dim,
            output_dim,
            head_dim,
            window_size,
            drop_path,
            type="SW",
            mlp_ratio=mlp_ratio,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = input_tensor.permute(0, 2, 3, 1).contiguous()
        output = self.block_1(output)
        output = self.block_2(output)
        return output.permute(0, 3, 1, 2).contiguous()


class ConvTransBlock(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        type: str = "W",
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if type not in {"W", "SW"}:
            raise ValueError(f"Unsupported attention type: {type}")

        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1)
        self.conv_block = ResidualBlock(conv_dim, conv_dim)
        self.trans_block = Block(
            trans_dim,
            trans_dim,
            head_dim,
            window_size,
            drop_path,
            type=type,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        mixed = self.conv1_1(input_tensor)
        conv_tensor, trans_tensor = torch.split(
            mixed,
            (self.conv_dim, self.trans_dim),
            dim=1,
        )
        conv_tensor = self.conv_block(conv_tensor) + conv_tensor
        trans_tensor = trans_tensor.permute(0, 2, 3, 1).contiguous()
        trans_tensor = self.trans_block(trans_tensor)
        trans_tensor = trans_tensor.permute(0, 3, 1, 2).contiguous()
        output = torch.cat((conv_tensor, trans_tensor), dim=1)
        return input_tensor + self.conv1_2(output)


class SWAtten(AttentionBlock):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        inter_dim: Optional[int] = 192,
    ) -> None:
        hidden_dim = inter_dim or input_dim
        super().__init__(N=hidden_dim)
        self.in_conv = (
            conv1x1(input_dim, hidden_dim) if inter_dim is not None else nn.Identity()
        )
        self.out_conv = (
            conv1x1(hidden_dim, output_dim) if inter_dim is not None else nn.Identity()
        )
        self.non_local_block = SwinBlock(
            hidden_dim,
            hidden_dim,
            head_dim,
            window_size,
            drop_path,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.in_conv(input_tensor)
        identity = output
        non_local = self.non_local_block(output)
        output = self.conv_a(output) * torch.sigmoid(self.conv_b(non_local))
        output = output + identity
        return self.out_conv(output)


class WinResidualUnit(nn.Module):
    """1x1 -> 3x3 -> 1x1 GELU residual unit; bottleneck width is half the
    input channels. Used inside :class:`WinNoShiftAttention`."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv1x1(channels, channels // 2),
            nn.GELU(),
            conv3x3(channels // 2, channels // 2),
            nn.GELU(),
            conv1x1(channels // 2, channels),
        )
        self.act = nn.GELU()

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.act(self.conv(input_tensor) + input_tensor)


class _WinBasedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        drop_path: float,
        output_proj: bool = True,
    ) -> None:
        super().__init__()
        attention_type = "SW" if shift_size > 0 else "W"
        self.attn = WMSA(
            input_dim=dim,
            output_dim=dim,
            head_dim=dim // num_heads,
            window_size=window_size,
            type=attention_type,
            output_proj=output_proj,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = input_tensor.permute(0, 2, 3, 1).contiguous()
        output = self.attn(output)
        output = output.permute(0, 3, 1, 2).contiguous()
        return input_tensor + self.drop_path(output)


class WinNoShiftAttention(nn.Module):
    """Sigmoid-gated dual-branch window attention block, used by STF / WACNN
    and (with ``output_proj=True``) by other window-attention CompressAI
    models. ``output_proj=False`` reproduces the STF / WACNN topology in which
    the WindowAttention output feeds straight back into the block without
    an additional Linear projection."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 0,
        drop_path: float = 0.0,
        output_proj: bool = True,
    ) -> None:
        super().__init__()
        self.conv_a = nn.Sequential(
            WinResidualUnit(dim),
            WinResidualUnit(dim),
            WinResidualUnit(dim),
        )
        self.conv_b = nn.Sequential(
            _WinBasedAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                drop_path=drop_path,
                output_proj=output_proj,
            ),
            WinResidualUnit(dim),
            WinResidualUnit(dim),
            WinResidualUnit(dim),
            conv1x1(dim, dim),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor + self.conv_a(input_tensor) * torch.sigmoid(
            self.conv_b(input_tensor)
        )


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, input_tensor: Tensor, height: int, width: int) -> Tensor:
        batch_size, length, channels = input_tensor.shape
        if length != height * width:
            raise ValueError("Input feature has wrong size.")

        output = input_tensor.view(batch_size, height, width, channels)
        if height % 2 == 1 or width % 2 == 1:
            output = F.pad(output, (0, 0, 0, width % 2, 0, height % 2))

        x0 = output[:, 0::2, 0::2, :]
        x1 = output[:, 1::2, 0::2, :]
        x2 = output[:, 0::2, 1::2, :]
        x3 = output[:, 1::2, 1::2, :]
        output = torch.cat([x0, x1, x2, x3], dim=-1)
        output = output.view(batch_size, -1, 4 * channels)
        return self.reduction(self.norm(output))


class PatchSplit(nn.Module):
    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.reduction = nn.Linear(dim, dim * 2, bias=False)
        self.norm = norm_layer(dim)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, input_tensor: Tensor, height: int, width: int) -> Tensor:
        batch_size, length, channels = input_tensor.shape
        if length != height * width:
            raise ValueError("Input feature has wrong size.")

        output = self.reduction(self.norm(input_tensor))
        output = output.permute(0, 2, 1).contiguous()
        output = output.view(batch_size, 2 * channels, height, width)
        output = self.shuffle(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        return output.view(batch_size, 4 * length, -1)


def __getattr__(name):
    if name == "Win_noShift_Attention":
        import warnings

        warnings.warn(
            "Win_noShift_Attention is deprecated; use WinNoShiftAttention instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return WinNoShiftAttention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
