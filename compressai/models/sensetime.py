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

import types

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import (
    AttentionBlock,
    CheckerboardMaskedConv2d,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv1x1,
    conv3x3,
    sequential_channel_ramp,
    subpel_conv3x3,
)
from compressai.registry import register_model

from .base import SimpleVAECompressionModel
from .utils import conv, deconv

__all__ = [
    "Cheng2020AnchorCheckerboard",
    "Elic2022Official",
    "Elic2022Chandelier",
]


@register_model("cheng2020-anchor-checkerboard")
class Cheng2020AnchorCheckerboard(SimpleVAECompressionModel):
    """Cheng2020 anchor model with checkerboard context model.

    Base transform model from [Cheng2020]. Context model from [He2021].

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(**kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                "y": CheckerboardLatentCodec(
                    latent_codec={
                        "y": GaussianConditionalLatentCodec(quantizer="ste"),
                    },
                    entropy_parameters=nn.Sequential(
                        nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
                    ),
                    context_prediction=CheckerboardMaskedConv2d(
                        N, 2 * N, kernel_size=5, stride=1, padding=2
                    ),
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


@register_model("elic2022-official")
class Elic2022Official(SimpleVAECompressionModel):
    """ELIC 2022; uneven channel groups with checkerboard spatial context.

    Context model from [He2022].
    Based on modified attention model architecture from [Cheng2020].

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    Args:
        N (int): Number of main network channels
        M (int): Number of latent space channels
        groups (list[int]): Number of channels in each channel group
    """

    def __init__(self, N=192, M=320, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N * 3 // 2, N * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


@register_model("elic2022-chandelier")
class Elic2022Chandelier(SimpleVAECompressionModel):
    """ELIC 2022; simplified context model using only first and most recent groups.

    Context model from [He2022], with simplifications and parameters
    from the [Chandelier2023] implementation.
    Based on modified attention model architecture from [Cheng2020].

    .. note::

        This implementation contains some differences compared to the
        original [He2022] paper. For instance, the implemented context
        model only uses the first and the most recently decoded channel
        groups to predict the current channel group. In contrast, the
        original paper uses all previously decoded channel groups.
        Also, the last layer of h_s is now a conv rather than a deconv.

    [Chandelier2023]: `"ELiC-ReImplemetation"
    <https://github.com/VincentChandelier/ELiC-ReImplemetation>`_, by
    Vincent Chandelier, 2023.

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    Args:
        N (int): Number of main network channels
        M (int): Number of latent space channels
        groups (list[int]): Number of channels in each channel group
    """

    def __init__(self, N=192, M=320, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N * 3 // 2, M * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": nn.Sequential(
                conv(
                    # Input: first group, and most recently decoded group.
                    self.groups[0] + (k > 1) * self.groups[k - 1],
                    224,
                    kernel_size=5,
                    stride=1,
                ),
                nn.ReLU(inplace=True),
                conv(224, 128, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(128, self.groups[k] * 2, kernel_size=5, stride=1),
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            nn.Sequential(
                conv1x1(
                    # Input: spatial context, channel context, and hyper params.
                    self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + M * 2,
                    M * 2,
                ),
                nn.ReLU(inplace=True),
                conv1x1(M * 2, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[k] * 2),
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(
                        quantizer="ste", chunks=("means", "scales")
                    ),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

        self._monkey_patch()

    def _monkey_patch(self):
        """Monkey-patch to use only first group and most recent group."""

        def merge_y(self: ChannelGroupsLatentCodec, *args):
            if len(args) == 0:
                return Tensor()
            if len(args) == 1:
                return args[0]
            if len(args) < len(self.groups):
                return torch.cat([args[0], args[-1]], dim=1)
            return torch.cat(args, dim=1)

        chan_groups_latent_codec = self.latent_codec["y"]
        obj = chan_groups_latent_codec
        obj.merge_y = types.MethodType(merge_y, obj)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block.

    Introduced by [He2016], this block sandwiches a 3x3 convolution
    between two 1x1 convolutions which reduce and then restore the
    number of channels. This reduces the number of parameters required.

    [He2016]: `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_, by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun, CVPR 2016.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out + identity
