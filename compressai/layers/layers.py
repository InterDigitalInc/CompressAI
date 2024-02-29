# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

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

import math

from typing import Any, Tuple

import torch
import torch.nn as nn

from torch import Tensor
from torch.autograd import Function

from .gdn import GDN

__all__ = [
    "AttentionBlock",
    "MaskedConv2d",
    "CheckerboardMaskedConv2d",
    "ResidualBlock",
    "ResidualBlockUpsample",
    "ResidualBlockWithStride",
    "conv1x1",
    "SpectralConv2d",
    "SpectralConvTranspose2d",
    "conv3x3",
    "subpel_conv3x3",
    "QReLU",
    "sequential_channel_ramp",
]


class _SpectralConvNdMixin:
    def __init__(self, dim: Tuple[int, ...]):
        self.dim = dim
        self.weight_transformed = nn.Parameter(self._to_transform_domain(self.weight))
        del self._parameters["weight"]  # Unregister weight, and fallback to property.

    @property
    def weight(self) -> Tensor:
        return self._from_transform_domain(self.weight_transformed)

    def _to_transform_domain(self, x: Tensor) -> Tensor:
        return torch.fft.rfftn(x, s=self.kernel_size, dim=self.dim, norm="ortho")

    def _from_transform_domain(self, x: Tensor) -> Tensor:
        return torch.fft.irfftn(x, s=self.kernel_size, dim=self.dim, norm="ortho")


class SpectralConv2d(nn.Conv2d, _SpectralConvNdMixin):
    r"""Spectral 2D convolution.

    Introduced in [Balle2018efficient].
    Reparameterizes the weights to be derived from weights stored in the
    frequency domain.
    In the original paper, this is referred to as "spectral Adam" or
    "Sadam" due to its effect on the Adam optimizer update rule.
    The motivation behind representing the weights in the frequency
    domain is that optimizer updates/steps may now affect all
    frequencies to an equal amount.
    This improves the gradient conditioning, thus leading to faster
    convergence and increased stability at larger learning rates.

    For comparison, see the TensorFlow Compression implementations of
    `SignalConv2D
    <https://github.com/tensorflow/compression/blob/v2.14.0/tensorflow_compression/python/layers/signal_conv.py#L61>`_
    and
    `RDFTParameter
    <https://github.com/tensorflow/compression/blob/v2.14.0/tensorflow_compression/python/layers/parameters.py#L71>`_.

    [Balle2018efficient]: `"Efficient Nonlinear Transforms for Lossy
    Image Compression" <https://arxiv.org/abs/1802.00847>`_,
    by Johannes Ballé, PCS 2018.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        _SpectralConvNdMixin.__init__(self, dim=(-2, -1))


class SpectralConvTranspose2d(nn.ConvTranspose2d, _SpectralConvNdMixin):
    r"""Spectral 2D transposed convolution.

    Transposed version of :class:`SpectralConv2d`.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        _SpectralConvNdMixin.__init__(self, dim=(-2, -1))


class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data = self.weight.data * self.mask
        return super().forward(x)


class CheckerboardMaskedConv2d(MaskedConv2d):
    r"""Checkerboard masked 2D convolution; mask future "unseen" pixels.

    Checkerboard mask variant used in
    `"Checkerboard Context Model for Efficient Learned Image Compression"
    <https://arxiv.org/abs/2103.15306>`_, by Dailan He, Yaoyan Zheng,
    Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        _, _, h, w = self.mask.size()
        self.mask[:] = 1
        self.mask[:, :, 0::2, 0::2] = 0
        self.mask[:, :, 1::2, 1::2] = 0
        self.mask[:, :, h // 2, w // 2] = mask_type == "B"


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class QReLU(Function):
    """QReLU

    Clamping input with given bit-depth range.
    Suppose that input data presents integer through an integer network
    otherwise any precision of input will simply clamp without rounding
    operation.

    Pre-computed scale with gamma function is used for backward computation.

    More details can be found in
    `"Integer networks for data compression with latent-variable models"
    <https://openreview.net/pdf?id=S1zz2i0cY7>`_,
    by Johannes Ballé, Nick Johnston and David Minnen, ICLR in 2019

    Args:
        input: a tensor data
        bit_depth: source bit-depth (used for clamping)
        beta: a parameter for modeling the gradient during backward computation
    """

    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        # TODO(choih): allow to use adaptive scale instead of
        # pre-computed scale with gamma function
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2**bit_depth - 1
        ctx.save_for_backward(input)

        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_sub = (
            torch.exp(
                (-ctx.alpha**ctx.beta)
                * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta
            )
            * grad_output.clone()
        )

        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]

        return grad_input, None, None


def sequential_channel_ramp(
    in_ch: int,
    out_ch: int,
    *,
    min_ch: int = 0,
    num_layers: int = 3,
    interp: str = "linear",
    make_layer=None,
    make_act=None,
    skip_last_act: bool = True,
    **layer_kwargs,
) -> nn.Module:
    """Interleave layers of gradually ramping channels with nonlinearities."""
    channels = ramp(in_ch, out_ch, num_layers + 1, method=interp).floor().int()
    channels[1:-1] = channels[1:-1].clip(min=min_ch)
    channels = channels.tolist()
    layers = [
        module
        for ch_in, ch_out in zip(channels[:-1], channels[1:])
        for module in [
            make_layer(ch_in, ch_out, **layer_kwargs),
            make_act(),
        ]
    ]
    if skip_last_act:
        layers = layers[:-1]
    return nn.Sequential(*layers)


def ramp(a, b, steps=None, method="linear", **kwargs):
    if method == "linear":
        return torch.linspace(a, b, steps, **kwargs)
    if method == "log":
        return torch.logspace(math.log10(a), math.log10(b), steps, **kwargs)
    raise ValueError(f"Unknown ramp method: {method}")
