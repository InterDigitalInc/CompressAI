import torch

import pytest

from compressai.layers import GDN, GDN1, MaskedConv2d


class TestScripting:
    def test_gdn(self):
        g = GDN(128)
        x = torch.rand(1, 128, 1, 1)
        y0 = g(x)

        m = torch.jit.script(g)
        y1 = m(x)

        assert torch.allclose(y0, y1)

    def test_gdn1(self):
        g = GDN1(128)
        x = torch.rand(1, 128, 1, 1)
        y0 = g(x)

        m = torch.jit.script(g)
        y1 = m(x)

        assert torch.allclose(y0, y1)

    def test_masked_conv_A(self):
        conv = MaskedConv2d(3, 3, 3, padding=1)

        with pytest.raises(RuntimeError):
            m = torch.jit.script(conv)
