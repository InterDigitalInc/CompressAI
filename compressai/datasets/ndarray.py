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

# Adapted via https://github.com/pytorch/pytorch/blob/v2.1.0/torch/utils/data/dataset.py
# BSD-style license: https://github.com/pytorch/pytorch/blob/v2.1.0/LICENSE

from typing import Tuple, Union

import numpy as np

from torch.utils.data import Dataset


class NdArrayDataset(Dataset[Union[np.ndarray, Tuple[np.ndarray, ...]]]):
    r"""Dataset wrapping arrays.

    Each sample will be retrieved by indexing arrays along the first dimension.

    Args:
        *arrays (np.ndarray): arrays that have the same size of the first dimension.
    """

    arrays: Tuple[np.ndarray, ...]

    def __init__(self, *arrays: np.ndarray, single: bool = False) -> None:
        assert all(
            arrays[0].shape[0] == array.shape[0] for array in arrays
        ), "Size mismatch between arrays"
        self.arrays = arrays
        self.single = single

    def __getitem__(self, index):
        if self.single:
            [array] = self.arrays
            return array[index]
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]
