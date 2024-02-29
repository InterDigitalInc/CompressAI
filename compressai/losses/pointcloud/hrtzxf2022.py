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

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers.pointcloud.hrtzxf2022 import index_points
from compressai.losses.utils import compute_rate_loss
from compressai.registry import register_criterion

from .chamfer import chamfer_distance


@register_criterion("RateDistortionLoss_hrtzxf2022")
class RateDistortionLoss_hrtzxf2022(nn.Module):
    """Loss introduced in [He2022pcc]_ for "hrtzxf2022-pcc-rec" model.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    LMBDA_DEFAULT = {
        "bpp": 1.0,
        "chamfer": 1e4,
        "chamfer_layers": (1.0, 0.1, 0.1),
        "latent_xyzs": 1e2,
        "mean_distance": 5e1,
        "normal": 1e2,
        "pts_num": 5e-3,
        "upsample_num": 1.0,
    }

    def __init__(
        self,
        lmbda=None,
        compress_normal=False,
        latent_xyzs_codec_mode="learned",
    ):
        super().__init__()
        self.lmbda = lmbda or dict(self.LMBDA_DEFAULT)
        self.compress_normal = compress_normal
        self.latent_xyzs_codec_mode = latent_xyzs_codec_mode

    def forward(self, output, target):
        device = target["pos"].device
        B, P, _ = target["pos"].shape

        out = {}

        chamfer_loss_, nearest_gt_idx_ = get_chamfer_loss(
            output["gt_xyz_"],
            output["xyz_hat_"],
        )

        out["chamfer_loss"] = sum(
            self.lmbda["chamfer_layers"][i] * chamfer_loss_[i]
            for i in range(len(chamfer_loss_))
        )

        out["rec_loss"] = chamfer_loss_[0]  # Name as rec_loss for compatibility.

        out["mean_distance_loss"], out["upsample_num_loss"] = get_density_loss(
            output["gt_downsample_num_"],
            output["gt_mean_distance_"],
            output["upsample_num_hat_"],
            output["mean_distance_hat_"],
            nearest_gt_idx_,
        )

        out["pts_num_loss"] = get_pts_num_loss(
            output["gt_xyz_"],
            output["upsample_num_hat_"],
        )

        if self.latent_xyzs_codec_mode == "learned":
            out["latent_xyzs_loss"] = get_latent_xyzs_loss(
                output["gt_latent_xyz"],
                output["latent_xyz_hat"],
            )
        elif self.latent_xyzs_codec_mode == "float16":
            out["latent_xyzs_loss"] = torch.tensor([0.0], device=device)
        else:
            raise ValueError(
                f"Unknown latent_xyzs_codec_mode: {self.latent_xyzs_codec_mode}"
            )

        if self.compress_normal:
            out["normal_loss"] = get_normal_loss(
                output["gt_normal"],
                output["feat_hat"].tanh(),
                nearest_gt_idx_[0],
            )
        else:
            out["normal_loss"] = torch.tensor([0.0], device=device)

        if "likelihoods" in output:
            out.update(compute_rate_loss(output["likelihoods"], B, P))

        out["loss"] = sum(
            self.lmbda[k] * out[f"{k}_loss"]
            for k in self.lmbda.keys()
            if f"{k}_loss" in out
        )

        return out


def get_chamfer_loss(gt_xyzs_, xyzs_hat_):
    num_layers = len(gt_xyzs_)
    chamfer_loss_ = []
    nearest_gt_idx_ = []

    for i in range(num_layers):
        xyzs1 = gt_xyzs_[i]
        xyzs2 = xyzs_hat_[num_layers - i - 1]
        dist1, dist2, _, idx2 = chamfer_distance(xyzs1, xyzs2, order="b c n")
        chamfer_loss_.append(dist1.mean() + dist2.mean())
        nearest_gt_idx_.append(idx2.long())

    return chamfer_loss_, nearest_gt_idx_


def get_density_loss(gt_dnums_, gt_mdis_, unums_hat_, mdis_hat_, nearest_gt_idx_):
    num_layers = len(gt_dnums_)
    l1_loss = nn.L1Loss(reduction="mean")
    mean_distance_loss_ = []
    upsample_num_loss_ = []

    for i in range(num_layers):
        if i == num_layers - 1:
            # At the final downsample layer, gt_latent_xyzs ≈ latent_xyzs_hat.
            mdis_i = gt_mdis_[i]
            dnum_i = gt_dnums_[i]
        else:
            idx = nearest_gt_idx_[i + 1]
            mdis_i = index_points(gt_mdis_[i].unsqueeze(1), idx).squeeze(1)
            dnum_i = index_points(gt_dnums_[i].unsqueeze(1), idx).squeeze(1)

        mean_distance_loss_.append(l1_loss(mdis_hat_[num_layers - i - 1], mdis_i))
        upsample_num_loss_.append(l1_loss(unums_hat_[num_layers - i - 1], dnum_i))

    return sum(mean_distance_loss_), sum(upsample_num_loss_)


def get_pts_num_loss(gt_xyzs_, unums_hat_):
    num_layers = len(gt_xyzs_)
    b, _, _ = gt_xyzs_[0].shape
    gt_num_points_ = [x.shape[2] for x in gt_xyzs_]
    return sum(
        torch.abs(unums_hat_[num_layers - i - 1].sum() - gt_num_points_[i] * b)
        for i in range(num_layers)
    )


def get_normal_loss(gt_normals, pred_normals, nearest_gt_idx):
    nearest_normal = index_points(gt_normals, nearest_gt_idx)
    return F.mse_loss(pred_normals, nearest_normal)


def get_latent_xyzs_loss(gt_latent_xyzs, latent_xyzs_hat):
    return F.mse_loss(gt_latent_xyzs, latent_xyzs_hat)
