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

import torch

import pytest

from compressai.layers import GDN, GDN1, MaskedConv2d


class TestMaskedConv2d:
    @staticmethod
    def test_mask_type():
        MaskedConv2d(1, 3, 3, mask_type='A')
        MaskedConv2d(1, 3, 3, mask_type='B')

        with pytest.raises(ValueError):
            MaskedConv2d(1, 3, 3, mask_type='C')

    @staticmethod
    def test_mask_A():
        conv = MaskedConv2d(1, 3, 5, mask_type='A')

        assert (conv.mask[0] == conv.mask[1]).all()
        assert (conv.mask[0] == conv.mask[2]).all()

        _, _, h, w = conv.mask.size()
        a = torch.ones_like(conv.mask)
        a[:, :, h // 2, w // 2:] = 0
        a[:, :, h // 2 + 1:] = 0

        assert (conv.mask == a).all()

    @staticmethod
    def test_mask_B():
        conv = MaskedConv2d(1, 3, 5, mask_type='B')

        assert (conv.mask[0] == conv.mask[1]).all()
        assert (conv.mask[0] == conv.mask[2]).all()

        _, _, h, w = conv.mask.size()
        b = torch.ones_like(conv.mask)
        b[:, :, h // 2, w // 2 + 1:] = 0
        b[:, :, h // 2 + 1:] = 0

        assert (conv.mask == b).all()

    @staticmethod
    def test_mask_A_1d():
        conv = MaskedConv2d(1, 3, (1, 5), mask_type='A')

        assert (conv.mask[0] == conv.mask[1]).all()
        assert (conv.mask[0] == conv.mask[2]).all()

        _, _, h, w = conv.mask.size()
        a = torch.ones_like(conv.mask)
        a[:, :, h // 2, w // 2:] = 0
        a[:, :, h // 2 + 1:] = 0

        assert (conv.mask == a).all()

    @staticmethod
    def test_mask_B_1d():
        conv = MaskedConv2d(3, 1, (5, 1), mask_type='B')

        assert (conv.mask[:, 0] == conv.mask[:, 1]).all()
        assert (conv.mask[:, 0] == conv.mask[:, 2]).all()

        _, _, h, w = conv.mask.size()
        b = torch.ones_like(conv.mask)
        b[:, :, h // 2, w // 2 + 1:] = 0
        b[:, :, h // 2 + 1:] = 0

        assert (conv.mask == b).all()

    @staticmethod
    def test_mask_multiple():
        cfgs = [
            # (in, out, kernel_size)
            (1, 3, 5),
            (3, 1, 3),
            (3, 3, 7),
        ]

        for cfg in cfgs:
            in_ch, out_ch, k = cfg
            conv = MaskedConv2d(in_ch, out_ch, k, mask_type='A')

            assert conv.mask[0].sum() != 0
            assert (conv.mask - conv.mask[0]).sum() == 0

            _, _, h, w = conv.mask.size()
            a = torch.ones_like(conv.mask)
            a[:, :, h // 2, w // 2:] = 0
            a[:, :, h // 2 + 1:] = 0

            assert (conv.mask == a).all()


class TestGDN:
    def test_gdn(self):
        g = GDN(32)
        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = g(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x / torch.sqrt(1 + .1 * (x**2))
        assert torch.allclose(y_ref, y)

    def test_igdn(self):
        g = GDN(32, inverse=True)
        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = g(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x * torch.sqrt(1 + .1 * (x**2))
        assert torch.allclose(y_ref, y)

    def test_gdn1(self):
        g = GDN1(32)
        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = g(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x / (1 + .1 * torch.abs(x))
        assert torch.allclose(y_ref, y)

    def test_igdn(self):
        g = GDN1(32, inverse=True)
        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = g(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x * (1 + .1 * torch.abs(x))
        assert torch.allclose(y_ref, y)
