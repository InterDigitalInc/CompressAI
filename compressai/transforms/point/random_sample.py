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


@functional_transform("random_sample")
@register_transform("RandomSample")
class RandomSample(BaseTransform):
    r"""Randomly samples points and associated attributes
    (functional name: :obj:`random_sample`).
    """

    def __init__(
        self,
        num=None,
        *,
        attrs=("pos",),
        remove_duplicates_by=None,
        preserve_order=False,
        seed=None,
        static_seed=None,
    ):
        self.num = num
        self.attrs = attrs
        self.remove_duplicates_by = remove_duplicates_by
        self.preserve_order = preserve_order
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.static_seed = static_seed

    def __call__(self, data: Data) -> Data:
        if self.static_seed is not None:
            self.generator.manual_seed(self.static_seed)

        if self.remove_duplicates_by is not None:
            _, perm = data[self.remove_duplicates_by].unique(return_inverse=True, dim=0)
            for attr in self.attrs:
                data[attr] = data[attr][perm]

        num_input = data[self.attrs[0]].shape[0]
        assert all(data[k].shape[0] == num_input for k in self.attrs)

        p = torch.ones(max(num_input, self.num), dtype=torch.float32)
        perm = torch.multinomial(p, self.num, generator=self.generator)
        perm %= num_input

        if self.preserve_order:
            perm = perm.sort()[0]

        return Data(**{k: v[perm] if k in self.attrs else v for k, v in data.items()})
