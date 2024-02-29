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

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from compressai.registry import register_transform


@functional_transform("random_rotate_full")
@register_transform("RandomRotateFull")
class RandomRotateFull(BaseTransform):
    r"""Randomly rotates node positions around the origin
    (functional name: :obj:`random_rotate_full`).
    """

    def __call__(self, data: Data) -> Data:
        _, ndim = data.pos.shape
        rot = random_rotation_matrix(1, ndim).to(data.pos.device).squeeze(0)
        data.pos = data.pos @ rot.T
        return data


# See https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/4832876#4832876
def random_rotation_matrix(batch_size: int, ndim=3, generator=None) -> torch.Tensor:
    z = torch.randn((batch_size, ndim, ndim), generator=generator)
    q, r = torch.linalg.qr(z)
    sign = 2 * (r.diagonal(dim1=-2, dim2=-1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[:, 0, :] *= torch.linalg.det(rot)[..., None]
    return rot
