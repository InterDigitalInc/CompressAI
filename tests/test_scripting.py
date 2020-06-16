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
