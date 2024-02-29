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

from __future__ import annotations

from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.layers.pointcloud.pointnet import (
    pointnet_g_a_simple,
    pointnet_g_s_simple,
)
from compressai.models import CompressionModel
from compressai.registry import register_model

__all__ = [
    "PointNetReconstructionPccModel",
]


@register_model("sfu2023-pcc-rec-pointnet")
class PointNetReconstructionPccModel(CompressionModel):
    """PointNet-based PCC reconstruction model.

    Model based on PointNet [Qi2017PointNet]_, modified for compression
    by [Yan2019]_, with layer configurations and other modifications as
    used in [Ulhaq2023]_.

    References:

        .. [Qi2017PointNet] `"PointNet: Deep Learning on Point Sets for
            3D Classification and Segmentation"
            <https://arxiv.org/abs/1612.00593>`_, by Charles R. Qi,
            Hao Su, Kaichun Mo, and Leonidas J. Guibas, CVPR 2017.

        .. [Yan2019] `"Deep AutoEncoder-based Lossy Geometry Compression
            for Point Clouds" <https://arxiv.org/abs/1905.03691>`_,
            by Wei Yan, Yiting Shao, Shan Liu, Thomas H Li, Zhu Li,
            and Ge Li, 2019.

        .. [Ulhaq2023] `"Learned Point Cloud Compression for
            Classification" <https://arxiv.org/abs/2308.05959>`_,
            by Mateen Ulhaq and Ivan V. Bajić, MMSP 2023.
    """

    latent_codec: LatentCodec

    def __init__(
        self,
        num_points=1024,
        num_channels={  # noqa: B006
            "g_a": [3, 64, 64, 64, 128, 1024],
            "g_s": [1024, 256, 512, 1024 * 3],
        },
        groups={  # noqa: B006
            "g_a": [1, 1, 1, 1, 1],
        },
    ):
        super().__init__()

        assert num_channels["g_a"][-1] == num_channels["g_s"][0]
        assert num_channels["g_s"][-1] == num_points * 3

        self.g_a = pointnet_g_a_simple(num_channels["g_a"], groups["g_a"])

        self.g_s = pointnet_g_s_simple(num_channels["g_s"])

        self.latent_codec = EntropyBottleneckLatentCodec(
            channels=num_channels["g_a"][-1],
            tail_mass=1e-4,
        )

    def forward(self, input):
        x = input["pos"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)
        assert x_hat.shape == x.shape

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            # Additional outputs:
            "y": y,
            "y_hat": y_hat,
            "debug_outputs": {
                "y_hat": y_hat,
            },
        }

    def compress(self, input):
        x = input["pos"]
        x_t = x.transpose(-2, -1)
        y = self.g_a(x_t)
        y_out = self.latent_codec.compress(y)
        [y_strings] = y_out["strings"]
        return {"strings": [y_strings], "shape": (1,)}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        [y_strings] = strings
        y_hat = self.latent_codec.decompress([y_strings], shape)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}
