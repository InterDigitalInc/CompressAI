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

from compressai.entropy_models import (EntropyBottleneck, EntropyModel,
                                       GaussianConditional)

from compressai.models.priors import FactorizedPrior


@pytest.fixture
def entropy_model():
    return EntropyModel()


class TestEntropyModel:
    def test_quantize_invalid(self, entropy_model):
        x = torch.rand(1, 3, 4, 4)
        with pytest.raises(ValueError):
            entropy_model._quantize(x, mode='toto')

    def test_quantize_noise(self, entropy_model):
        x = torch.rand(1, 3, 4, 4)
        y = entropy_model._quantize(x, 'noise')

        assert y.shape == x.shape
        assert ((y - x) <= 0.5).all()
        assert ((y - x) >= -0.5).all()
        assert (y != torch.round(x)).any()

    def test_quantize_symbols(self, entropy_model):
        x = torch.rand(1, 3, 4, 4)
        y = entropy_model._quantize(x, 'symbols')

        assert y.shape == x.shape
        assert (y == torch.round(x).int()).all()

    def test_quantize_dequantize(self, entropy_model):
        x = torch.rand(1, 3, 4, 4)
        means = torch.rand(1, 3, 4, 4)
        y = entropy_model._quantize(x, 'dequantize', means)

        assert y.shape == x.shape
        assert (y == torch.round(x - means) + means).all()

    def test_dequantize(self, entropy_model):
        x = torch.randint(-32, 32, (1, 3, 4, 4))
        means = torch.rand(1, 3, 4, 4)
        y = entropy_model._dequantize(x, means)

        assert y.shape == x.shape
        assert y.type() == means.type()

    def test_forward(self, entropy_model):
        with pytest.raises(NotImplementedError):
            entropy_model()

    def test_invalid_coder(self):
        with pytest.raises(ValueError):
            entropy_model = EntropyModel(entropy_coder='huffman')

        with pytest.raises(ValueError):
            entropy_model = EntropyModel(entropy_coder=0xff)

    def test_invalid_inputs(self, entropy_model):
        with pytest.raises(TypeError):
            entropy_model.compress(torch.rand(1, 3))
        with pytest.raises(ValueError):
            entropy_model.compress(torch.rand(1, 3), torch.rand(2, 3))
        with pytest.raises(ValueError):
            entropy_model.compress(torch.rand(1, 3, 1, 1), torch.rand(2, 3))

    def test_invalid_cdf(self, entropy_model):
        x = torch.rand(1, 32, 16, 16)
        indexes = torch.rand(1, 32, 16, 16)
        with pytest.raises(ValueError):
            entropy_model.compress(x, indexes)

    def test_invalid_cdf_length(self, entropy_model):
        x = torch.rand(1, 32, 16, 16)
        indexes = torch.rand(1, 32, 16, 16)
        entropy_model._quantized_cdf.resize_(32, 1)

        with pytest.raises(ValueError):
            entropy_model.compress(x, indexes)

        entropy_model._cdf_length.resize_(32, 1)
        with pytest.raises(ValueError):
            entropy_model.compress(x, indexes)

    def test_invalid_offsets(self, entropy_model):
        x = torch.rand(1, 32, 16, 16)
        indexes = torch.rand(1, 32, 16, 16)
        entropy_model._quantized_cdf.resize_(32, 1)
        entropy_model._cdf_length.resize_(32)
        with pytest.raises(ValueError):
            entropy_model.compress(x, indexes)

    def test_invalid_decompress(self, entropy_model):
        with pytest.raises(TypeError):
            entropy_model.decompress(['ssss'])

        with pytest.raises(ValueError):
            entropy_model.decompress('sss', torch.rand(1, 3, 4, 4))

        with pytest.raises(ValueError):
            entropy_model.decompress(['sss'], torch.rand(1, 4, 4))

        with pytest.raises(ValueError):
            entropy_model.decompress(['sss'], torch.rand(2, 4, 4))

        with pytest.raises(ValueError):
            entropy_model.decompress(['sss'], torch.rand(1, 4, 4),
                                     torch.rand(2, 4, 4))


class TestEntropyBottleneck:
    def test_forward_training(self):
        entropy_bottleneck = EntropyBottleneck(128)
        x = torch.rand(1, 128, 32, 32)
        y, y_likelihoods = entropy_bottleneck(x)

        assert isinstance(entropy_bottleneck, EntropyModel)
        assert y.shape == x.shape
        assert y_likelihoods.shape == x.shape

        assert ((y - x) <= 0.5).all()
        assert ((y - x) >= -0.5).all()
        assert (y != torch.round(x)).any()

    def test_forward_inference(self):
        entropy_bottleneck = EntropyBottleneck(128)
        entropy_bottleneck.eval()
        x = torch.rand(1, 128, 32, 32)
        y, y_likelihoods = entropy_bottleneck(x)

        assert y.shape == x.shape
        assert y_likelihoods.shape == x.shape

        assert (y == torch.round(x)).all()

    def test_loss(self):
        entropy_bottleneck = EntropyBottleneck(128)
        loss = entropy_bottleneck.loss()

        assert len(loss.size()) == 0
        assert loss.numel() == 1

    def test_scripting(self):
        entropy_bottleneck = EntropyBottleneck(128)
        x = torch.rand(1, 128, 32, 32)

        torch.manual_seed(32)
        y0 = entropy_bottleneck(x)

        m = torch.jit.script(entropy_bottleneck)

        torch.manual_seed(32)
        y1 = m(x)

        assert torch.allclose(y0[0], y1[0])
        assert torch.all(y1[1] == 0)  # not yet supported


class TestGaussianConditional:
    def test_invalid_scale_table(self):
        with pytest.raises(ValueError):
            GaussianConditional(1)

        with pytest.raises(ValueError):
            GaussianConditional([])

        with pytest.raises(ValueError):
            GaussianConditional(())

        with pytest.raises(ValueError):
            GaussianConditional(torch.rand(10))

        with pytest.raises(ValueError):
            GaussianConditional([2, 1])

        with pytest.raises(ValueError):
            GaussianConditional([0, 1, 2])

        with pytest.raises(ValueError):
            GaussianConditional([], scale_bound=None)

    def test_forward_training(self):
        gaussian_conditional = GaussianConditional(None)
        x = torch.rand(1, 128, 32, 32)
        scales = torch.rand(1, 128, 32, 32)
        y, y_likelihoods = gaussian_conditional(x, scales)

        assert isinstance(gaussian_conditional, EntropyModel)
        assert y.shape == x.shape
        assert y_likelihoods.shape == x.shape

        assert ((y - x) <= 0.5).all()
        assert ((y - x) >= -0.5).all()
        assert (y != torch.round(x)).any()

    def test_forward_inference(self):
        gaussian_conditional = GaussianConditional(None)
        gaussian_conditional.eval()
        x = torch.rand(1, 128, 32, 32)
        scales = torch.rand(1, 128, 32, 32)
        y, y_likelihoods = gaussian_conditional(x, scales)

        assert y.shape == x.shape
        assert y_likelihoods.shape == x.shape

        assert (y == torch.round(x)).all()

    def test_forward_training_mean(self):
        gaussian_conditional = GaussianConditional(None)
        x = torch.rand(1, 128, 32, 32)
        scales = torch.rand(1, 128, 32, 32)
        means = torch.rand(1, 128, 32, 32)
        y, y_likelihoods = gaussian_conditional(x, scales, means)

        assert y.shape == x.shape
        assert y_likelihoods.shape == x.shape

        assert ((y - x) <= 0.5).all()
        assert ((y - x) >= -0.5).all()
        assert (y != torch.round(x)).any()

    def test_forward_inference_mean(self):
        gaussian_conditional = GaussianConditional(None)
        gaussian_conditional.eval()
        x = torch.rand(1, 128, 32, 32)
        scales = torch.rand(1, 128, 32, 32)
        means = torch.rand(1, 128, 32, 32)
        y, y_likelihoods = gaussian_conditional(x, scales, means)

        assert y.shape == x.shape
        assert y_likelihoods.shape == x.shape

        assert (y == torch.round(x - means) + means).all()

    def test_scripting(self):
        gaussian_conditional = GaussianConditional(None)
        x = torch.rand(1, 128, 32, 32)
        scales = torch.rand(1, 128, 32, 32)
        means = torch.rand(1, 128, 32, 32)

        torch.manual_seed(32)
        y0 = gaussian_conditional(x, scales, means)

        m = torch.jit.script(gaussian_conditional)

        torch.manual_seed(32)
        y1 = m(x, scales, means)

        assert torch.allclose(y0[0], y1[0])
        assert torch.allclose(y0[1], y1[1])
