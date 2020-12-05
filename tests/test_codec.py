# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

import pytest
import torch

from compressai.zoo import models

archs = [
    "bmshj2018-factorized",
    "bmshj2018-hyperprior",
    "mbt2018-mean",
    "mbt2018",
]


class TestCompressDecompress:
    @pytest.mark.parametrize("arch,N", itertools.product(archs, [1, 2, 3]))
    def test_codec(self, arch: str, N: int):
        x = torch.zeros(N, 3, 256, 256)
        h, w = x.size()[-2:]
        x[:, :, h // 4 : -h // 4, w // 4 : -w // 4].fill_(1)

        model = models[arch]
        net = model(quality=1, metric="mse", pretrained=True).eval()
        with torch.no_grad():
            rv = net.compress(x)

        assert "shape" in rv
        shape = rv["shape"]
        ds = net.downsampling_factor
        assert shape == torch.Size([x.size(2) // ds, x.size(3) // ds])

        assert "strings" in rv
        strings_list = rv["strings"]
        # y_strings (+ optional z_strings)
        assert len(strings_list) == 1 or len(strings_list) == 2
        for strings in strings_list:
            assert len(strings) == N
            for string in strings:
                assert isinstance(string, bytes)
                assert len(string) > 0

        with torch.no_grad():
            rv = net.decompress(strings_list, shape)
        assert "x_hat" in rv
        x_hat = rv["x_hat"]
        assert x_hat.size() == x.size()

        mse = torch.mean((x - x_hat) ** 2)
        psnr = -10 * torch.log10(mse).item()
        assert 35 < psnr < 41
