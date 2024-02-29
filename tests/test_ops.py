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

import pytest
import torch

from compressai._CXX import pmf_to_quantized_cdf
from compressai.ops import LowerBound, NonNegativeParametrizer, quantize_ste


class TestQuantizeSTE:
    def test_quantize_ste_ok(self):
        x = torch.rand(16)
        assert (quantize_ste(x) == torch.round(x)).all()

    def test_quantize_ste_grads(self):
        x = torch.rand(24, requires_grad=True)
        y = quantize_ste(x)
        y.backward(x)
        assert x.grad is not None
        assert (x.grad == x).all()


class TestLowerBound:
    def test_lower_bound_ok(self):
        x = torch.rand(16)
        bound = torch.rand(1)
        lower_bound = LowerBound(bound)
        assert (lower_bound(x) == torch.max(x, bound)).all()

    def test_lower_bound_script(self):
        x = torch.rand(16)
        bound = torch.rand(1)
        lower_bound = LowerBound(bound)
        scripted = torch.jit.script(lower_bound)
        assert (scripted(x) == torch.max(x, bound)).all()

    def test_lower_bound_grads(self):
        x = torch.rand(16, requires_grad=True)
        bound = torch.rand(1)
        lower_bound = LowerBound(bound)
        y = lower_bound(x)
        y.backward(x)

        assert x.grad is not None
        assert (x.grad == ((x >= bound) * x)).all()


class TestNonNegativeParametrizer:
    def test_non_negative(self):
        parametrizer = NonNegativeParametrizer()
        x = torch.rand(1, 8, 8, 8) * 2 - 1  # [0, 1] -> [-1, 1]
        x_reparam = parametrizer(x)

        assert x_reparam.shape == x.shape
        assert x_reparam.min() >= 0

    def test_non_negative_init(self):
        parametrizer = NonNegativeParametrizer()
        x = torch.rand(1, 8, 8, 8) * 2 - 1
        x_init = parametrizer.init(x)

        assert x_init.shape == x.shape
        assert torch.allclose(x_init, torch.sqrt(torch.max(x, x - x)), atol=2**-18)

    def test_non_negative_min(self):
        for _ in range(10):
            minimum = torch.rand(1)
            parametrizer = NonNegativeParametrizer(minimum.item())
            x = torch.rand(1, 8, 8, 8) * 2 - 1
            x_reparam = parametrizer(x)

            assert x_reparam.shape == x.shape
            assert torch.allclose(x_reparam.min(), minimum)


class TestPmfToQuantizedCDF:
    def test_ok(self):
        out = pmf_to_quantized_cdf([0.1, 0.2, 0, 0], 16)
        assert out == [0, 21845, 65534, 65535, 65536]

    def test_negative_prob(self):
        with pytest.raises(ValueError):
            pmf_to_quantized_cdf([1, 0, -1], 16)

    @pytest.mark.parametrize("v", ("inf", "-inf", "nan"))
    def test_non_finite_prob(self, v):
        with pytest.raises(ValueError):
            pmf_to_quantized_cdf([1, 0, float(v)], 16)

        with pytest.raises(ValueError):
            pmf_to_quantized_cdf([1, 0, float(v), 2, 3, 4], 16)
