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

from compressai.ops import (LowerBound, NonNegativeParametrizer, ste_round)


class TestSTERound:
    def test_ste_round_ok(self):
        x = torch.rand(16)
        assert (ste_round(x) == torch.round(x)).all()

    def test_ste_round_grads(self):
        x = torch.rand(24, requires_grad=True)
        y = ste_round(x)
        y.backward(x)
        assert x.grad is not None
        assert (x.grad == x).all()


class TestLowerBound:
    def test_lower_bound_ok(self):
        x = torch.rand(16)
        bound = torch.rand(1)
        lower_bound = LowerBound(bound)

        assert (lower_bound(x) == torch.max(x, bound)).all()

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
        assert torch.allclose(x_init,
                              torch.sqrt(torch.max(x, x - x)),
                              atol=2**-18)

    def test_non_negative_min(self):
        for _ in range(10):
            minimum = torch.rand(1)
            parametrizer = NonNegativeParametrizer(minimum.item())
            x = torch.rand(1, 8, 8, 8) * 2 - 1
            x_reparam = parametrizer(x)

            assert x_reparam.shape == x.shape
            assert torch.allclose(x_reparam.min(), minimum)
