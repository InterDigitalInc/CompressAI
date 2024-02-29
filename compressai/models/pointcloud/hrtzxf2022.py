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

# Code adapted from https://github.com/yunhe20/D-PCC

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.layers.pointcloud.hrtzxf2022 import (
    DownsampleLayer,
    EdgeConv,
    RefineLayer,
    UpsampleLayer,
    UpsampleNumLayer,
    nearby_distance_sum,
)
from compressai.layers.pointcloud.utils import select_xyzs_and_feats
from compressai.models import CompressionModel
from compressai.registry import register_model

__all__ = [
    "DensityPreservingReconstructionPccModel",
]


@register_model("hrtzxf2022-pcc-rec")
class DensityPreservingReconstructionPccModel(CompressionModel):
    """Density-preserving deep point cloud compression.

    Model introduced by [He2022pcc]_.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(
        self,
        downsample_rate=(1 / 3, 1 / 3, 1 / 3),
        candidate_upsample_rate=(8, 8, 8),
        in_dim=3,
        feat_dim=8,
        hidden_dim=64,
        k=16,
        ngroups=1,
        sub_point_conv_mode="mlp",
        compress_normal=False,
        latent_xyzs_codec=None,
        **kwargs,
    ):
        super().__init__()

        self.compress_normal = compress_normal

        self.pre_conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            nn.GroupNorm(ngroups, hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, feat_dim, 1),
        )

        self.encoder = Encoder(
            downsample_rate,
            feat_dim,
            hidden_dim,
            k,
            ngroups,
        )

        self.decoder = Decoder(
            downsample_rate,
            candidate_upsample_rate,
            feat_dim,
            hidden_dim,
            k,
            sub_point_conv_mode,
            compress_normal,
        )

        self.latent_codec = nn.ModuleDict(
            {
                "feat": EntropyBottleneckLatentCodec(channels=feat_dim),
                "xyz": XyzsLatentCodec(
                    feat_dim, hidden_dim, k, ngroups, **(latent_xyzs_codec or {})
                ),
            }
        )

    def _prepare_input(self, input):
        input_data = [input["pos"]]
        if self.compress_normal:
            input_data.append(input["normal"])
        input_data = torch.cat(input_data, dim=1).permute(0, 2, 1).contiguous()

        xyzs = input_data[:, :3].contiguous()
        gt_normals = input_data[:, 3 : 3 + 3 * self.compress_normal].contiguous()
        feats = input_data

        return xyzs, gt_normals, feats

    def forward(self, input):
        # xyzs: (b, 3, n)

        xyzs, gt_normals, feats = self._prepare_input(input)

        feats = self.pre_conv(feats)

        gt_xyzs_, gt_dnums_, gt_mdis_, latent_xyzs, latent_feats = self.encoder(
            xyzs, feats
        )

        gt_latent_xyzs = latent_xyzs

        # NOTE: Temporarily reshape to (b, c, m, 1) for compatibility.
        latent_feats = latent_feats.unsqueeze(-1)
        latent_feats_out = self.latent_codec["feat"](latent_feats)
        latent_feats_hat = latent_feats_out["y_hat"].squeeze(-1)

        latent_xyzs_out = self.latent_codec["xyz"](latent_xyzs)
        latent_xyzs_hat = latent_xyzs_out["y_hat"]

        xyzs_hat_, unums_hat_, mdis_hat_, feats_hat = self.decoder(
            latent_xyzs_hat, latent_feats_hat
        )

        # Permute final xyzs_hat back to (b, n, c)
        xyzs_hat = xyzs_hat_[-1].permute(0, 2, 1).contiguous()

        return {
            "x_hat": xyzs_hat,
            "xyz_hat_": xyzs_hat_,
            "latent_xyz_hat": latent_xyzs_hat,
            "feat_hat": feats_hat,
            "upsample_num_hat_": unums_hat_,
            "mean_distance_hat_": mdis_hat_,
            "gt_xyz_": gt_xyzs_,
            "gt_latent_xyz": gt_latent_xyzs,
            "gt_normal": gt_normals,
            "gt_downsample_num_": gt_dnums_,
            "gt_mean_distance_": gt_mdis_,
            "likelihoods": {
                "latent_feat": latent_feats_out["likelihoods"]["y"],
                "latent_xyz": latent_xyzs_out["likelihoods"]["y"],
            },
        }

    def compress(self, input):
        xyzs, _, feats = self._prepare_input(input)

        feats = self.pre_conv(feats)

        _, _, _, latent_xyzs, latent_feats = self.encoder(xyzs, feats)

        latent_feats = latent_feats.unsqueeze(-1)
        latent_feats_out = self.latent_codec["feat"].compress(latent_feats)

        latent_xyzs = latent_xyzs
        latent_xyzs_out = self.latent_codec["xyz"].compress(latent_xyzs)

        return {
            "strings": [
                latent_feats_out["strings"],
                latent_xyzs_out["strings"],
            ],
            "shape": [
                latent_feats_out["shape"],
                latent_xyzs_out["shape"],
            ],
        }

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        latent_feats_out = self.latent_codec["feat"].decompress(strings[0], shape[0])
        latent_feats_hat = latent_feats_out["y_hat"].squeeze(-1)

        latent_xyzs_out = self.latent_codec["xyz"].decompress(strings[1], shape[1])
        latent_xyzs_hat = latent_xyzs_out["y_hat"]

        xyzs_hat_, _, _, feats_hat = self.decoder(latent_xyzs_hat, latent_feats_hat)

        # Permute final xyzs_hat back to (b, n, c)
        xyzs_hat = xyzs_hat_[-1].permute(0, 2, 1).contiguous()

        return {
            "x_hat": xyzs_hat,
            "feat_hat": feats_hat,
        }


class XyzsLatentCodec(nn.Module):
    def __init__(self, dim, hidden_dim, k, ngroups, mode="learned", conv_mode="mlp"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            if conv_mode == "edge_conv":
                self.analysis = EdgeConv(3, dim, hidden_dim, k)
                self.synthesis = EdgeConv(dim, 3, hidden_dim, k)
            elif conv_mode == "mlp":
                self.analysis = nn.Sequential(
                    nn.Conv1d(3, hidden_dim, 1),
                    nn.GroupNorm(ngroups, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_dim, dim, 1),
                )
                self.synthesis = nn.Sequential(
                    nn.Conv1d(dim, hidden_dim, 1),
                    nn.GroupNorm(ngroups, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_dim, 3, 1),
                )
            else:
                raise ValueError(f"Unknown conv_mode: {conv_mode}")
            self.entropy_bottleneck = EntropyBottleneck(dim)
        else:
            self.placeholder = nn.Parameter(torch.empty(0))

    def forward(self, latent_xyzs):
        if self.mode == "learned":
            z = self.analysis(latent_xyzs)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            latent_xyzs_hat = self.synthesis(z_hat)
        elif self.mode == "float16":
            z_likelihoods = latent_xyzs.new_full(latent_xyzs.shape, 2**-16)
            latent_xyzs_hat = latent_xyzs.to(torch.float16).float()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return {"likelihoods": {"y": z_likelihoods}, "y_hat": latent_xyzs_hat}

    def compress(self, latent_xyzs):
        if self.mode == "learned":
            z = self.analysis(latent_xyzs)
            shape = z.shape[2:]
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
            latent_xyzs_hat = self.synthesis(z_hat)
        elif self.mode == "float16":
            z = latent_xyzs
            shape = z.shape[2:]
            z_hat = latent_xyzs.to(torch.float16)
            z_strings = [
                np.ascontiguousarray(x, dtype=">f2").tobytes()
                for x in z_hat.cpu().numpy()
            ]
            latent_xyzs_hat = z_hat.float()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return {"strings": [z_strings], "shape": shape, "y_hat": latent_xyzs_hat}

    def decompress(self, strings, shape):
        [z_strings] = strings
        if self.mode == "learned":
            z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
            latent_xyzs_hat = self.synthesis(z_hat)
        elif self.mode == "float16":
            z_hat = [np.frombuffer(s, dtype=">f2").reshape(shape) for s in z_strings]
            z_hat = torch.from_numpy(np.stack(z_hat)).to(self.placeholder.device)
            latent_xyzs_hat = z_hat.float()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return {"y_hat": latent_xyzs_hat}


class Encoder(nn.Module):
    def __init__(self, downsample_rate, dim, hidden_dim, k, ngroups):
        super().__init__()
        downsample_layers = [
            DownsampleLayer(downsample_rate[i], dim, hidden_dim, k, ngroups)
            for i in range(len(downsample_rate))
        ]
        self.downsample_layers = nn.ModuleList(downsample_layers)

    def forward(self, xyzs, feats):
        # xyzs: (b, 3, n)
        # feats: (b, c, n)

        gt_xyzs_ = []
        gt_dnums_ = []
        gt_mdis_ = []

        for downsample_layer in self.downsample_layers:
            gt_xyzs_.append(xyzs)
            xyzs, feats, downsample_num, mean_distance = downsample_layer(xyzs, feats)
            gt_dnums_.append(downsample_num)
            gt_mdis_.append(mean_distance)

        latent_xyzs = xyzs
        latent_feats = feats

        return gt_xyzs_, gt_dnums_, gt_mdis_, latent_xyzs, latent_feats


class Decoder(nn.Module):
    def __init__(
        self,
        downsample_rate,
        candidate_upsample_rate,
        dim,
        hidden_dim,
        k,
        sub_point_conv_mode,
        compress_normal,
    ):
        super().__init__()

        self.k = k
        self.compress_normal = compress_normal
        self.num_layers = len(downsample_rate)
        self.downsample_rate = downsample_rate

        self.upsample_layers = nn.ModuleList(
            [
                UpsampleLayer(
                    dim,
                    hidden_dim,
                    k,
                    sub_point_conv_mode,
                    candidate_upsample_rate[i],
                )
                for i in range(self.num_layers)
            ]
        )

        self.upsample_num_layers = nn.ModuleList(
            [
                UpsampleNumLayer(
                    dim,
                    hidden_dim,
                    candidate_upsample_rate[i],
                )
                for i in range(self.num_layers)
            ]
        )

        self.refine_layers = nn.ModuleList(
            [
                RefineLayer(
                    dim,
                    hidden_dim,
                    k,
                    sub_point_conv_mode,
                    compress_normal and i == self.num_layers - 1,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, xyzs, feats):
        # xyzs: (b, 3, n)
        # feats: (b, c, n)

        latent_xyzs = xyzs.clone()

        xyzs_hat_ = []
        unums_hat_ = []

        for i, (upsample_nn, upsample_num_nn, refine_nn) in enumerate(
            zip(self.upsample_layers, self.upsample_num_layers, self.refine_layers)
        ):
            # candidate_xyzs: (b, 3, n u)
            # candidate_feats: (b, c, n u)
            # upsample_num: (b, n)
            # xyzs: (b, 3, m)  [after upsample and select]
            # feats: (b, c, m) [after upsample and select]

            # For each point within the current set of "n" points,
            # upsample a fixed number "u" of candidate points.
            # The resulting candidate points have the shape (n, u).
            candidate_xyzs, candidate_feats = upsample_nn(xyzs, feats)

            # Determine local point cloud density near each upsampled group:
            upsample_num = upsample_num_nn(feats)

            # Subsample each point group to match the desired local density.
            # That is, from the i-th point group, select upsample_num[..., i] points.
            # Then, collect all the points so the resulting point set has shape (m_i,).
            #
            # If the batch size is >1, then the "m_i"s may be different.
            # In that case, resample each point set within the batch
            # until they all have the same shape (m,).
            # This can be done by either selecting a subset or
            # duplicating points as necessary.
            #
            # Since one of the goals is to reduce local point cloud
            # density in certain regions, we are happy with throwing
            # away distinct points, and then duplicating the remaining
            # points until they can fit within the desired tensor shape.

            # Select subset of points to match predicted local point cloud densities:
            xyzs, feats = select_xyzs_and_feats(
                candidate_xyzs,
                candidate_feats,
                upsample_num,
                upsample_rate=1 / self.downsample_rate[self.num_layers - i - 1],
            )

            # Refine upsampled points.
            xyzs, feats = refine_nn(xyzs, feats)

            xyzs_hat_.append(xyzs)
            unums_hat_.append(upsample_num)

        # Compute mean distance between centroids and the upsampled points.
        mdis_hat_ = self.get_pred_mdis([latent_xyzs, *xyzs_hat_], unums_hat_)

        return xyzs_hat_, unums_hat_, mdis_hat_, feats

    def get_pred_mdis(self, xyzs_hat_, unums_hat_):
        mdis_hat_ = []

        for prev_xyzs, curr_xyzs, curr_unums in zip(
            xyzs_hat_[:-1], xyzs_hat_[1:], unums_hat_
        ):
            # Compute mean distance for each point in "prev" to upsampled "curr".
            distance, _, _, _ = nearby_distance_sum(prev_xyzs, curr_xyzs, self.k)
            curr_mdis = distance / curr_unums
            mdis_hat_.append(curr_mdis)

        return mdis_hat_
