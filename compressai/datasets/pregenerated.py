# Copyright (c) 2021-2022, InterDigital Communications, Inc
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

from pathlib import Path
from typing import Tuple, Union

import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from compressai.registry import register_dataset

_size_2_t = Union[int, Tuple[int, int]]


@register_dataset("PreGeneratedMemmapDataset")
class PreGeneratedMemmapDataset(Dataset):
    """A data loader for memory-mapped numpy arrays.

    This allows for fast training where the images patches have already been
    extracted and shuffled. The numpy array in expected to have the following
    size: `NxHxWx3`, with `N` the number of samples, `H` and `W` the images
    dimensions.

    Args:
        root (string): root directory where the numpy arrays are located.
        image_size (int, int): size of the images in the array.
        patch_size (int): size of the patches to be randomly cropped for training.
        split (string): split mode ('train' or 'val').
        batch_size (int): batch size.
        num_workers (int): number of CPU thread workers.
        pin_memory (bool): pin memory.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        split: str = "train",
        image_size: _size_2_t = (256, 256),
    ):
        if not Path(root).is_dir():
            raise RuntimeError(f"Invalid path {root}")

        self.split = split
        self.transform = transform

        self.shuffle = False

        if split == "train":
            filename = "training.npy"
        elif split == "valid":
            filename = "validation.npy"
        else:
            raise ValueError()
        path = Path(root) / filename
        data: np.ndarray = np.memmap(path, mode="r", dtype="uint8")
        assert data.size > 0
        image_size = _coerce_size_2_t(image_size)
        self.data = data.reshape((-1, image_size[0], image_size[1], 3))

    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.fromarray(sample)
        if self.transform:
            return self.transform(sample)
        return sample

    def __len__(self):
        return self.data.shape[0]


def _coerce_size_2_t(x: _size_2_t) -> Tuple[int, int]:
    if isinstance(x, int):
        return x, x
    return x
