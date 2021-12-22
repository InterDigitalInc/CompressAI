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

import pytest
import torch

from PIL import Image
from torchvision import transforms

from compressai.datasets import ImageFolder


def save_fake_image(filepath, size=(512, 512)):
    img = Image.new("RGB", size=size)
    img.save(filepath)


class TestImageFolder:
    def test_init_ok(self, tmpdir):
        tmpdir.mkdir("train")
        tmpdir.mkdir("test")

        train_dataset = ImageFolder(tmpdir, split="train")
        test_dataset = ImageFolder(tmpdir, split="test")

        assert len(train_dataset) == 0
        assert len(test_dataset) == 0

    def test_count_ok(self, tmpdir):
        tmpdir.mkdir("train")
        (tmpdir / "train" / "img1.jpg").write("")
        (tmpdir / "train" / "img2.jpg").write("")
        (tmpdir / "train" / "img3.jpg").write("")

        train_dataset = ImageFolder(tmpdir, split="train")

        assert len(train_dataset) == 3

    def test_invalid_dir(self, tmpdir):
        with pytest.raises(RuntimeError):
            ImageFolder(tmpdir)

    def test_load(self, tmpdir):
        tmpdir.mkdir("train")
        save_fake_image((tmpdir / "train" / "img0.jpeg").strpath)

        train_dataset = ImageFolder(tmpdir, split="train")
        assert isinstance(train_dataset[0], Image.Image)

    def test_load_transforms(self, tmpdir):
        tmpdir.mkdir("train")
        save_fake_image((tmpdir / "train" / "img0.jpeg").strpath)

        transform = transforms.Compose(
            [
                transforms.CenterCrop((128, 128)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = ImageFolder(tmpdir, split="train", transform=transform)
        assert isinstance(train_dataset[0], torch.Tensor)
        assert train_dataset[0].size() == (3, 128, 128)
