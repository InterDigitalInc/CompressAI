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

from pathlib import Path
from typing import Tuple

import pytest
import torch

from PIL import Image
from torchvision import transforms

from compressai.datasets.video import VideoFolder


def save_fake_video(videopath: Path, size: Tuple = (512, 512)):
    frames = [Image.new("RGB", size=size) for _ in range(3)]
    for i in range(3):
        frames[i].save(f"{videopath}/img_{i}.jpg")


class TestVideoFolder:
    def test_init_ok(self, tmpdir):
        tmpdir.mkdir("sequences")
        size = (512, 512)
        train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomCrop(size)]
        )
        test_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(size)]
        )
        trainsplitfile = Path(f"{tmpdir}/train.list")
        testsplitfile = Path(f"{tmpdir}/test.list")
        with open(trainsplitfile, "w") as f:
            f.write("")
        with open(testsplitfile, "w") as f:
            f.write("")

        train_dataset = VideoFolder(tmpdir, split="train", transform=train_transforms)
        test_dataset = VideoFolder(tmpdir, split="test", transform=test_transforms)

        assert len(train_dataset) == 0
        assert len(test_dataset) == 0

    def test_count_ok(self, tmpdir):
        tmpdir.mkdir("sequences")
        tmpdir.mkdir("sequences/vid0")
        tmpdir.mkdir("sequences/vid1")
        size = (512, 512)
        trainsplitfile = Path(f"{tmpdir}/train.list")
        # testsplitfile = Path(f"{tmpdir}/test.list")
        # splitdir = Path(f"{tmpdir}/sequences")

        with open(trainsplitfile, "w") as f:
            f.write("vid0\nvid1")

        #   continue oco
        (tmpdir / "sequences" / "vid0" / "vid0_0.jpg").write("")
        (tmpdir / "sequences" / "vid0" / "vid0_1.jpg").write("")
        (tmpdir / "sequences" / "vid1" / "vid1_0.jpg").write("")

        train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomCrop(size)]
        )
        train_dataset = VideoFolder(tmpdir, split="train", transform=train_transforms)

        assert len(train_dataset) == 2

    def test_invalid_dir(self, tmpdir):
        with pytest.raises(RuntimeError):
            VideoFolder(tmpdir)

    def test_load(self, tmpdir):
        size = (512, 512)
        tmpdir.mkdir("sequences")
        tmpdir.mkdir("sequences/vid0")
        trainsplitfile = Path(f"{tmpdir}/train.list")
        with open(trainsplitfile, "w") as f:
            f.write("vid0\n")
        save_fake_video((tmpdir / "sequences" / "vid0").strpath)
        train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomCrop(size)]
        )
        train_dataset = VideoFolder(tmpdir, split="train", transform=train_transforms)
        print(type(train_dataset[0][0]))
        assert isinstance(train_dataset[0][0], torch.Tensor)
