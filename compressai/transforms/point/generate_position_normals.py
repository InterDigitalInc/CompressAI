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

from contextlib import suppress

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from compressai.registry import register_transform


@functional_transform("generate_position_normals")
@register_transform("GeneratePositionNormals")
class GeneratePositionNormals(BaseTransform):
    r"""Generates normals from node positions
    (functional name: :obj:`generate_position_normals`).
    """

    def __init__(self, *, method="any", **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        assert data.pos.ndim == 2 and data.pos.shape[1] == 3

        if self.method == "open3d":
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data.pos.cpu().numpy())
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN())
            pcd.normalize_normals()
            data.norm = torch.tensor(
                pcd.normals, dtype=torch.float32, device=data.pos.device
            )

            return data

        if self.method == "pytorch3d":
            import pytorch3d.ops

            data.norm = pytorch3d.ops.estimate_pointcloud_normals(
                data.pos.unsqueeze(0), **self.kwargs
            ).squeeze(0)

            return data

        if self.method == "any":
            for self.method in ["open3d", "pytorch3d"]:
                with suppress(ImportError):
                    return self(data)
            raise RuntimeError("Please install open3d / pytorch3d to estimate normals.")

        raise ValueError(f"Unknown method: {self.method}")
