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

from einops import rearrange

try:
    from pointops.functions import pointops
except ImportError:
    pass  # NOTE: Optional dependency.

from compressai.layers.pointcloud.hrtzxf2022 import index_points
from compressai.losses.utils import compute_rate_loss
from compressai.registry import register_criterion


@register_criterion("ChamferPccRateDistortionLoss")
class ChamferPccRateDistortionLoss(nn.Module):
    """Simple loss for regular point cloud compression.

    For compression models that reconstruct the input point cloud.
    """

    LMBDA_DEFAULT = {
        # "bpp": 1.0,
        "rec": 1.0,
    }

    def __init__(self, lmbda=None, rate_key="bpp"):
        super().__init__()
        self.lmbda = lmbda or dict(self.LMBDA_DEFAULT)
        self.lmbda.setdefault(rate_key, 1.0)

    def forward(self, output, target):
        out = {
            **self.compute_rate_loss(output, target),
            **self.compute_rec_loss(output, target),
        }

        out["loss"] = sum(
            self.lmbda[k] * out[f"{k}_loss"]
            for k in self.lmbda.keys()
            if f"{k}_loss" in out
        )

        return out

    def compute_rate_loss(self, output, target):
        if "likelihoods" not in output:
            return {}
        N, P, _ = target["pos"].shape
        return compute_rate_loss(output["likelihoods"], N, P)

    def compute_rec_loss(self, output, target):
        dist1, dist2, _, _ = chamfer_distance(
            target["pos"], output["x_hat"], order="b n c"
        )
        loss_chamfer = dist1.mean() + dist2.mean()
        return {"rec_loss": loss_chamfer}


def chamfer_distance(xyzs1, xyzs2, order="b n c"):
    # idx1, dist1: (b, n1)
    # idx2, dist2: (b, n2)
    xyzs1_bcn = rearrange(xyzs1, f"{order} -> b c n").contiguous()
    xyzs1_bnc = rearrange(xyzs1, f"{order} -> b n c").contiguous()
    xyzs2_bcn = rearrange(xyzs2, f"{order} -> b c n").contiguous()
    xyzs2_bnc = rearrange(xyzs2, f"{order} -> b n c").contiguous()
    idx1 = pointops.knnquery_heap(1, xyzs2_bnc, xyzs1_bnc).long().squeeze(2)
    idx2 = pointops.knnquery_heap(1, xyzs1_bnc, xyzs2_bnc).long().squeeze(2)
    torch.cuda.empty_cache()
    dist1 = ((xyzs1_bcn - index_points(xyzs2_bcn, idx1)) ** 2).sum(1)
    dist2 = ((xyzs2_bcn - index_points(xyzs1_bcn, idx2)) ** 2).sum(1)
    return dist1, dist2, idx1, idx2
