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
