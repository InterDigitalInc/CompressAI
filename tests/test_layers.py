# Copyright (c) 2021-2025, InterDigital Communications, Inc
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

from compressai.layers import (
    GDN,
    GDN1,
    AttentionBlock,
    MaskedConv2d,
    QReLU,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)


class TestMaskedConv2d:
    @staticmethod
    def test_mask_type():
        MaskedConv2d(1, 3, 3, mask_type="A")
        MaskedConv2d(1, 3, 3, mask_type="B")

        with pytest.raises(ValueError):
            MaskedConv2d(1, 3, 3, mask_type="C")

    @staticmethod
    def test_mask_A():
        conv = MaskedConv2d(1, 3, 5, mask_type="A")

        assert (conv.mask[0] == conv.mask[1]).all()
        assert (conv.mask[0] == conv.mask[2]).all()

        _, _, h, w = conv.mask.size()
        a = torch.ones_like(conv.mask)
        a[:, :, h // 2, w // 2 :] = 0
        a[:, :, h // 2 + 1 :] = 0

        assert (conv.mask == a).all()

    @staticmethod
    def test_mask_B():
        conv = MaskedConv2d(1, 3, 5, mask_type="B")

        assert (conv.mask[0] == conv.mask[1]).all()
        assert (conv.mask[0] == conv.mask[2]).all()

        _, _, h, w = conv.mask.size()
        b = torch.ones_like(conv.mask)
        b[:, :, h // 2, w // 2 + 1 :] = 0
        b[:, :, h // 2 + 1 :] = 0

        assert (conv.mask == b).all()

    @staticmethod
    def test_mask_A_1d():
        conv = MaskedConv2d(1, 3, (1, 5), mask_type="A")

        assert (conv.mask[0] == conv.mask[1]).all()
        assert (conv.mask[0] == conv.mask[2]).all()

        _, _, h, w = conv.mask.size()
        a = torch.ones_like(conv.mask)
        a[:, :, h // 2, w // 2 :] = 0
        a[:, :, h // 2 + 1 :] = 0

        assert (conv.mask == a).all()

    @staticmethod
    def test_mask_B_1d():
        conv = MaskedConv2d(3, 1, (5, 1), mask_type="B")

        assert (conv.mask[:, 0] == conv.mask[:, 1]).all()
        assert (conv.mask[:, 0] == conv.mask[:, 2]).all()

        _, _, h, w = conv.mask.size()
        b = torch.ones_like(conv.mask)
        b[:, :, h // 2, w // 2 + 1 :] = 0
        b[:, :, h // 2 + 1 :] = 0

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
            conv = MaskedConv2d(in_ch, out_ch, k, mask_type="A")

            assert conv.mask[0].sum() != 0
            assert (conv.mask - conv.mask[0]).sum() == 0

            _, _, h, w = conv.mask.size()
            a = torch.ones_like(conv.mask)
            a[:, :, h // 2, w // 2 :] = 0
            a[:, :, h // 2 + 1 :] = 0

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

        y_ref = x / torch.sqrt(1 + 0.1 * (x**2))
        assert torch.allclose(y_ref, y)

    def test_igdn(self):
        g = GDN(32, inverse=True)
        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = g(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x * torch.sqrt(1 + 0.1 * (x**2))
        assert torch.allclose(y_ref, y)

    def test_gdn1(self):
        g = GDN1(32)
        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = g(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x / (1 + 0.1 * torch.abs(x))
        assert torch.allclose(y_ref, y)


def test_ResidualBlockWithStride():
    layer = ResidualBlockWithStride(32, 64, stride=1)
    layer(torch.rand(1, 32, 4, 4))

    layer = ResidualBlockWithStride(32, 32, stride=1)
    layer(torch.rand(1, 32, 4, 4))

    layer = ResidualBlockWithStride(32, 32, stride=2)
    layer(torch.rand(1, 32, 4, 4))

    layer = ResidualBlockWithStride(32, 64, stride=2)
    layer(torch.rand(1, 32, 4, 4))


def test_ResidualBlockUpsample():
    layer = ResidualBlockUpsample(8, 16)
    layer(torch.rand(1, 8, 4, 4))


def test_ResidualBlock():
    layer = ResidualBlock(8, 8)
    layer(torch.rand(1, 8, 4, 4))

    layer = ResidualBlock(8, 16)
    layer(torch.rand(1, 8, 4, 4))


def test_AttentionBlock():
    layer = AttentionBlock(8)
    layer(torch.rand(1, 8, 4, 4))


class TestMultiScaleDictionaryCrossAttentionGLU:
    @staticmethod
    def test_forward_shape():
        from compressai.layers.attn.dictionary import (
            MultiScaleDictionaryCrossAttentionGLU,
        )

        mod = MultiScaleDictionaryCrossAttentionGLU(
            input_dim=192,
            output_dim=320,
            head_num=4,
            dictionary_dim=128,
        )
        x = torch.randn(2, 192, 4, 4)
        dictionary = torch.randn(2, 16, 128)
        out = mod(x, dictionary)
        assert out.shape == (2, 320, 4, 4)

    @staticmethod
    def test_state_dict_round_trip():
        from compressai.layers.attn.dictionary import (
            MultiScaleDictionaryCrossAttentionGLU,
        )

        mod = MultiScaleDictionaryCrossAttentionGLU(
            input_dim=192,
            output_dim=320,
            head_num=4,
            dictionary_dim=128,
        )
        mod2 = MultiScaleDictionaryCrossAttentionGLU(
            input_dim=192,
            output_dim=320,
            head_num=4,
            dictionary_dim=128,
        )
        mod2.load_state_dict(mod.state_dict(), strict=True)
        x = torch.randn(2, 192, 4, 4)
        dictionary = torch.randn(2, 16, 128)
        assert torch.allclose(mod(x, dictionary), mod2(x, dictionary))

    @staticmethod
    def test_dictionary_dim_default_matches_head_num():
        from compressai.layers.attn.dictionary import (
            MultiScaleDictionaryCrossAttentionGLU,
        )

        # Default dictionary_dim = 32 * head_num
        mod = MultiScaleDictionaryCrossAttentionGLU(
            input_dim=64, output_dim=128, head_num=4
        )
        x = torch.randn(1, 64, 2, 2)
        dictionary = torch.randn(1, 8, 128)
        out = mod(x, dictionary)
        assert out.shape == (1, 128, 2, 2)

    @staticmethod
    def test_dictionary_dim_must_divide_head_num():
        from compressai.layers.attn.dictionary import (
            MultiScaleDictionaryCrossAttentionGLU,
        )

        with pytest.raises(ValueError, match="divisible"):
            MultiScaleDictionaryCrossAttentionGLU(
                input_dim=32, output_dim=64, head_num=3, dictionary_dim=128
            )


class TestWavelet:
    @staticmethod
    def test_dwt_idwt_round_trip():
        pytest.importorskip("pytorch_wavelets")
        from compressai.layers.wave import DWT2D, IDWT2D

        dwt = DWT2D(wave="haar")
        idwt = IDWT2D(wave="haar")
        x = torch.randn(2, 3, 16, 16)
        sub = dwt(x)
        # 4 subbands -> output channels = 4 * input
        assert sub.shape == (2, 12, 8, 8)
        rec = idwt(sub)
        assert rec.shape == x.shape
        assert (rec - x).abs().max().item() < 1e-5

    @staticmethod
    def test_is_pytorch_wavelets_available_returns_bool():
        from compressai.layers.wave import is_pytorch_wavelets_available

        assert isinstance(is_pytorch_wavelets_available(), bool)


class TestQReLU:
    @staticmethod
    def test_QReLU():
        def qrelu(input, bit_depth=8, beta=100):
            return QReLU.apply(input, bit_depth, beta)

        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = qrelu(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x.clamp(min=0, max=2**8 - 1)
        assert torch.allclose(y_ref, y)
