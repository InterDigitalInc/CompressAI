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

import torch

from PIL import Image
from torch.utils.data import Dataset

from compressai.registry import register_dataset


@register_dataset("Vimeo90kDataset")
class Vimeo90kDataset(Dataset):
    """Load a Vimeo-90K structured dataset.

    Vimeo-90K dataset from
    Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, William T. Freeman:
    `"Video Enhancement with Task-Oriented Flow"
    <https://arxiv.org/abs/1711.09078>`_,
    International Journal of Computer Vision (IJCV), 2019.

    Training and testing image samples are respectively stored in
    separate directories:

    .. code-block::

        - rootdir/
            - sequence/
                - 00001/001/im1.png
                - 00001/001/im2.png
                - 00001/001/im3.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function for image/sequence transformation
        transform_frame (callable, optional): a function for frame transformation
        split (string): split mode ('train' or 'valid')
        tuplet (int): order of dataset tuplet (e.g. 3 for "triplet" dataset)
        mode (string): item grouping mode ('image' or 'video'). If 'image', each
            item is a single frame. If 'video', each item is a sequence of frames.
    """

    TUPLET_PREFIX = {3: "tri", 7: "sep"}
    SPLIT_TO_LIST_SUFFIX = {"train": "trainlist", "valid": "testlist"}

    def __init__(
        self,
        root,
        transform=None,
        transform_frame=None,
        split="train",
        tuplet=3,
        mode="image",
    ):
        self.mode = mode
        self.tuplet = tuplet

        list_path = Path(root) / self._list_filename(split, tuplet)

        with open(list_path) as f:
            self.sequences = [
                f"{root}/sequences/{line.rstrip()}" for line in f if line.strip() != ""
            ]

        self.frames = [
            f"{seq}/im{idx}.png"
            for seq in self.sequences
            for idx in range(1, tuplet + 1)
        ]

        self.transform = transform
        self.transform_frame = transform_frame  # Suggested: transforms.ToTensor()

    def __getitem__(self, index):
        if self.mode == "image":
            item = self._get_frame(self.frames[index])
        elif self.mode == "video":
            item = torch.stack(
                [
                    self._get_frame(f"{self.sequences[index]}/im{idx}.png")
                    for idx in range(1, self.tuplet + 1)
                ]
            )
        else:
            raise ValueError(f"Invalid mode {self.mode}. Must be 'image' or 'video'.")
        if self.transform:
            item = self.transform(item)
        return item

    def _get_frame(self, filename):
        frame = Image.open(filename).convert("RGB")
        if self.transform_frame:
            frame = self.transform_frame(frame)
        return frame

    def __len__(self):
        if self.mode == "image":
            return len(self.frames)
        elif self.mode == "video":
            return len(self.sequences)
        else:
            raise ValueError(f"Invalid mode {self.mode}. Must be 'image' or 'video'.")

    def _list_filename(self, split: str, tuplet: int) -> str:
        tuplet_prefix = self.TUPLET_PREFIX[tuplet]
        list_suffix = self.SPLIT_TO_LIST_SUFFIX[split]
        return f"{tuplet_prefix}_{list_suffix}.txt"
