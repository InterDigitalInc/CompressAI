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

import itertools

import pytest
import torch

from compressai.zoo import image_models

archs = [
    "bmshj2018-factorized",
    "bmshj2018-hyperprior",
    "mbt2018-mean",
    "mbt2018",
]


class TestCompressDecompress:
    @pytest.mark.parametrize("arch,N", itertools.product(archs, [1, 2, 3]))
    def test_image_codec(self, arch: str, N: int):
        x = torch.zeros(N, 3, 256, 256)
        h, w = x.size()[-2:]
        x[:, :, h // 4 : -h // 4, w // 4 : -w // 4].fill_(1)

        model = image_models[arch]
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
