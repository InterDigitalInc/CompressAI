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

import torch
import torch.nn as nn

from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.layers.basic import Gain, Interleave, Reshape, Transpose
from compressai.layers.pointcloud.pointnet import GAIN
from compressai.layers.pointcloud.pointnet2 import PointNetSetAbstraction
from compressai.layers.pointcloud.pointnet2_sfu import UpsampleBlock
from compressai.models import CompressionModel
from compressai.registry import register_model

__all__ = [
    "PointNet2SsgReconstructionPccModel",
]


@register_model("sfu2024-pcc-rec-pointnet2-ssg")
class PointNet2SsgReconstructionPccModel(CompressionModel):
    """PointNet++-based PCC reconstruction model.

    Model based on PointNet++ [Qi2017PointNetPlusPlus]_, and modified
    for compression by [Ulhaq2024]_.
    Uses single-scale grouping (SSG) for point set abstraction.

    References:

        .. [Qi2017PointNetPlusPlus] `"PointNet++: Deep Hierarchical
            Feature Learning on Point Sets in a Metric Space"
            <https://arxiv.org/abs/1706.02413>`_, by Charles R. Qi,
            Li Yi, Hao Su, and Leonidas J. Guibas, NIPS 2017.

        .. [Ulhaq2024] `"Scalable Human-Machine Point Cloud Compression"
            <TODO>`_,
            by Mateen Ulhaq and Ivan V. Bajić, PCS 2024.
    """

    def __init__(
        self,
        num_points=1024,
        num_classes=40,
        D=(0, 128, 192, 256),
        P=(1024, 256, 64, 1),
        S=(None, 4, 4, 64),
        R=(None, 0.2, 0.4, None),
        E=(3, 64, 32, 16, 0),
        M=(0, 0, 64, 64),
        normal_channel=False,
    ):
        """
        Args:
            num_points: Number of input points. [unused]
            num_classes: Number of object classes. [unused]
            D: Number of input feature channels.
            P: Number of output points.
            S: Number of samples per centroid.
            R: Radius of the ball to query points within.
            E: Number of output feature channels after each upsample.
            M: Number of latent channels for the bottleneck.
            normal_channel: Whether the input includes normals.
        """
        super().__init__()

        self.num_points = num_points
        self.num_classes = num_classes
        self.D = D
        self.P = P
        self.S = S
        self.R = R
        self.E = E
        self.M = M
        self.normal_channel = bool(normal_channel)

        # Original PointNet++ architecture:
        # D = [3 * self.normal_channel, 128, 256, 1024]
        # P = [None, 512, 128, 1]
        # S = [None, 32, 64, 128]
        # R = [None, 0.2, 0.4, None]

        # NOTE: P[0] is only used to determine the number of output points.
        # assert P[0] == num_points

        assert P[0] == P[1] * S[1]
        assert P[1] == P[2] * S[2]
        assert P[2] == P[3] * S[3]

        self.levels = 4

        self.down = nn.ModuleDict(
            {
                "_1": PointNetSetAbstraction(
                    npoint=P[1],
                    radius=R[1],
                    nsample=S[1],
                    in_channel=D[0] + 3,
                    mlp=[D[1] // 2, D[1] // 2, D[1]],
                    group_all=False,
                ),
                "_2": PointNetSetAbstraction(
                    npoint=P[2],
                    radius=R[2],
                    nsample=S[2],
                    in_channel=D[1] + 3,
                    mlp=[D[1], D[1], D[2]],
                    group_all=False,
                ),
                "_3": PointNetSetAbstraction(
                    npoint=None,
                    radius=None,
                    nsample=None,
                    in_channel=D[2] + 3,
                    mlp=[D[2], D[3], D[3]],
                    group_all=True,
                ),
            }
        )

        i_final = self.levels - 1
        groups_h_final = 1 if D[i_final] * M[i_final] <= 2**16 else 4

        self.h_a = nn.ModuleDict(
            {
                **{
                    f"_{i}": nn.Sequential(
                        Reshape((D[i] + 3, P[i + 1] * S[i + 1])),
                        nn.Conv1d(D[i] + 3, M[i], 1),
                        Gain((M[i], 1), factor=GAIN),
                    )
                    for i in range(self.levels - 1)
                    if M[i] > 0
                },
                f"_{i_final}": nn.Sequential(
                    Reshape((D[i_final], 1)),
                    nn.Conv1d(D[i_final], M[i_final], 1, groups=groups_h_final),
                    Interleave(groups=groups_h_final),
                    Gain((M[i_final], 1), factor=GAIN),
                ),
            }
        )

        self.h_s = nn.ModuleDict(
            {
                **{
                    f"_{i}": nn.Sequential(
                        Gain((M[i], 1), factor=1 / GAIN),
                        nn.Conv1d(M[i], D[i] + 3, 1),
                    )
                    for i in range(self.levels - 1)
                    if M[i] > 0
                },
                f"_{i_final}": nn.Sequential(
                    Gain((M[i_final], 1), factor=1 / GAIN),
                    nn.Conv1d(M[i_final], D[i_final], 1, groups=groups_h_final),
                    Interleave(groups=groups_h_final),
                ),
            }
        )

        self.up = nn.ModuleDict(
            {
                "_0": nn.Sequential(
                    nn.Conv1d(E[1] + D[0] + 3 * bool(M[0]), E[1], 1),
                    # nn.BatchNorm1d(E[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(E[1], E[0], 1),
                    Reshape((E[0], P[0])),
                    Transpose(-2, -1),
                ),
                "_1": UpsampleBlock(D, E, M, P, S, i=1, extra_in_ch=3, groups=(1, 4)),
                "_2": UpsampleBlock(D, E, M, P, S, i=2, extra_in_ch=3, groups=(1, 4)),
                "_3": UpsampleBlock(D, E, M, P, S, i=3, extra_in_ch=0, groups=(1, 4)),
            }
        )

        self.latent_codec = nn.ModuleDict(
            {
                f"_{i}": EntropyBottleneckLatentCodec(channels=M[i], tail_mass=1e-4)
                for i in range(self.levels)
                if M[i] > 0
            }
        )

    def forward(self, input):
        xyz, norm = self._get_inputs(input)
        y_out_, u_, uu_ = self._compress(xyz, norm, mode="forward")
        x_hat, y_hat_, v_ = self._decompress(y_out_, mode="forward")

        return {
            "x_hat": x_hat,
            "likelihoods": {
                f"y_{i}": y_out_[i]["likelihoods"]["y"]
                for i in range(self.levels)
                if "likelihoods" in y_out_[i]
            },
            # Additional outputs:
            "debug_outputs": {
                **{f"u_{i}": v for i, v in u_.items() if v is not None},
                **{f"uu_{i}": v for i, v in uu_.items()},
                **{f"y_hat_{i}": v for i, v in y_hat_.items()},
                **{f"v_{i}": v for i, v in v_.items() if v.numel() > 0},
            },
        }

    def compress(self, input):
        xyz, norm = self._get_inputs(input)
        y_out_, _, _ = self._compress(xyz, norm, mode="compress")

        return {
            # "strings": {f"y_{i}": y_out_[i]["strings"] for i in range(self.levels)},
            # Flatten nested structure into list[list[str]]:
            "strings": [
                ss for level in range(self.levels) for ss in y_out_[level]["strings"]
            ],
            "shape": {f"y_{i}": y_out_[i]["shape"] for i in range(self.levels)},
        }

    def decompress(self, strings, shape):
        y_inputs_ = {
            i: {
                "strings": [strings[i]],
                "shape": shape[f"y_{i}"],
            }
            for i in range(self.levels)
        }

        x_hat, _, _ = self._decompress(y_inputs_, mode="decompress")

        return {
            "x_hat": x_hat,
        }

    def _get_inputs(self, input):
        points = input["pos"].transpose(-2, -1)
        if self.normal_channel:
            xyz = points[:, :3, :]
            norm = points[:, 3:, :]
        else:
            xyz = points
            norm = None
        return xyz, norm

    def _compress(self, xyz, norm, *, mode):
        lc_func = {"forward": lambda lc: lc, "compress": lambda lc: lc.compress}[mode]

        B, _, _ = xyz.shape

        xyz_ = {0: xyz}
        u_ = {0: norm}
        uu_ = {}
        y_ = {}
        y_out_ = {}

        for i in range(1, self.levels):
            down_out_i = self.down[f"_{i}"](xyz_[i - 1], u_[i - 1])
            xyz_[i] = down_out_i["new_xyz"]
            u_[i] = down_out_i["new_features"]
            uu_[i - 1] = down_out_i["grouped_features"]

        uu_[self.levels - 1] = u_[self.levels - 1][:, :, None, :]

        for i in reversed(range(0, self.levels)):
            if self.M[i] == 0:
                y_out_[i] = {"strings": [[b""] * B], "shape": ()}
                continue

            y_[i] = self.h_a[f"_{i}"](uu_[i])
            # NOTE: Reshape 1D -> 2D since latent codecs expect 2D inputs.
            y_out_[i] = lc_func(self.latent_codec[f"_{i}"])(y_[i][..., None])

        return y_out_, u_, uu_

    def _decompress(self, y_inputs_, *, mode):
        y_hat_ = {}
        y_out_ = {}
        uu_hat_ = {}
        v_ = {}

        for i in reversed(range(0, self.levels)):
            if self.M[i] == 0:
                continue
            if mode == "forward":
                y_out_[i] = y_inputs_[i]
            elif mode == "decompress":
                y_out_[i] = self.latent_codec[f"_{i}"].decompress(
                    y_inputs_[i]["strings"], shape=y_inputs_[i]["shape"]
                )
            # NOTE: Reshape 2D -> 1D since latent codecs return 2D outputs.
            y_hat_[i] = y_out_[i]["y_hat"].squeeze(-1)
            uu_hat_[i] = self.h_s[f"_{i}"](y_hat_[i])

        B, _, *tail = uu_hat_[self.levels - 1].shape
        v_[self.levels] = uu_hat_[self.levels - 1].new_zeros((B, 0, *tail))

        for i in reversed(range(0, self.levels)):
            v_[i] = self.up[f"_{i}"](
                v_[i + 1]
                if self.M[i] == 0
                else torch.cat([v_[i + 1], uu_hat_[i]], dim=1)
            )

        x_hat = v_[0]

        return x_hat, y_hat_, v_
