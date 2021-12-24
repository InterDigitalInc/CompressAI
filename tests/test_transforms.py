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

from compressai.transforms import RGB2YCbCr, YCbCr2RGB, YUV420To444, YUV444To420
from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)


@pytest.mark.parametrize("func", (rgb2ycbcr, ycbcr2rgb))
def test_invalid_input(func):
    with pytest.raises(ValueError):
        func(torch.rand(1, 3).numpy())

    with pytest.raises(ValueError):
        func(torch.rand(1, 3))

    with pytest.raises(ValueError):
        func(torch.rand(1, 4, 4, 4))

    with pytest.raises(ValueError):
        func(torch.rand(1, 3, 4, 4).int())


@pytest.mark.parametrize("func", (rgb2ycbcr, ycbcr2rgb))
def test_ok(func):
    x = torch.rand(1, 3, 32, 32)
    rv = func(x)
    assert rv.size() == x.size()
    assert rv.type() == x.type()

    x = torch.rand(3, 64, 64)
    rv = func(x)
    assert rv.size() == x.size()
    assert rv.type() == x.type()


def test_round_trip():
    x = torch.rand(1, 3, 32, 32)
    rv = ycbcr2rgb(rgb2ycbcr(x))
    assert torch.allclose(x, rv, atol=1e-5)

    rv = rgb2ycbcr(ycbcr2rgb(x))
    assert torch.allclose(x, rv, atol=1e-5)


def test_444_to_420():
    x = torch.rand(1, 3, 32, 32)
    y, u, v = yuv_444_to_420(x)

    assert u.size(0) == v.size(0) == y.size(0) == x.size(0)
    assert u.size(1) == v.size(1) == y.size(1) == 1
    assert y.size(2) == x.size(2) and y.size(3) == x.size(3)
    assert u.size(2) == v.size(2) == (y.size(2) // 2)
    assert u.size(3) == v.size(3) == (y.size(2) // 2)

    assert (x[:, [0]] == y).all()

    with pytest.raises(ValueError):
        y, u, v = yuv_444_to_420(x, mode="toto")

    y, u, v = yuv_444_to_420(x.chunk(3, 1))


def test_420_to_444():
    y = torch.rand(1, 1, 32, 32)
    u = torch.rand(1, 1, 16, 16)
    v = torch.rand(1, 1, 16, 16)

    with pytest.raises(ValueError):
        yuv_420_to_444((y, u))

    with pytest.raises(ValueError):
        yuv_420_to_444((y, u, v), mode="bilateral")

    rv = yuv_420_to_444((y, u, v))
    assert isinstance(rv, torch.Tensor)
    assert (rv[:, [0]] == y).all()

    rv = yuv_420_to_444((y, u, v), return_tuple=True)
    assert all(isinstance(c, torch.Tensor) for c in rv)
    assert (rv[0] == y).all()
    assert rv[0].size() == rv[1].size() == rv[2].size()


def test_transforms():
    x = torch.rand(1, 3, 32, 32)
    rv = RGB2YCbCr()(x)
    assert rv.size() == x.size()
    repr(RGB2YCbCr())

    rv = YCbCr2RGB()(x)
    assert rv.size() == x.size()
    repr(YCbCr2RGB())

    rv = YUV444To420()(x)
    assert len(rv) == 3
    repr(YUV444To420())

    rv = YUV420To444()(rv)
    assert rv.size() == x.size()
    repr(YUV420To444())
