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

import copy

import pytest
import torch

from compressai.entropy_models import (
    EntropyBottleneck,
    EntropyModel,
    GaussianConditional,
)
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior


@pytest.fixture
def entropy_model():
    return EntropyModel()


class TestEntropyModel:
    def test_quantize_invalid(self, entropy_model):
        x = torch.rand(1, 3, 4, 4)
        with pytest.raises(ValueError):
            entropy_model.quantize(x, mode="toto")

    def test_quantize_noise(self, entropy_model):
        x = torch.rand(1, 3, 4, 4)
        y = entropy_model.quantize(x, "noise")

        assert y.shape == x.shape
        assert ((y - x) <= 0.5).all()
        assert ((y - x) >= -0.5).all()
        assert (y != torch.round(x)).any()

    def test__quantize(self, entropy_model):
        x = torch.rand(1, 3, 4, 4)
        s = torch.rand(1).item()
        torch.manual_seed(s)
        y0 = entropy_model.quantize(x, "noise")
        torch.manual_seed(s)

        with pytest.warns(UserWarning):
            y1 = entropy_model._quantize(x, "noise")
        assert (y0 == y1).all()

    def test_quantize_symbols(self, entropy_model):
        x = torch.rand(1, 3, 4, 4)
        y = entropy_model.quantize(x, "symbols")

        assert y.shape == x.shape
        assert (y == torch.round(x).int()).all()

    def test_quantize_dequantize(self, entropy_model):
        x = torch.rand(1, 3, 4, 4)
        means = torch.rand(1, 3, 4, 4)
        y = entropy_model.quantize(x, "dequantize", means)

        assert y.shape == x.shape
        assert (y == torch.round(x - means) + means).all()

    def test_dequantize(self, entropy_model):
        x = torch.randint(-32, 32, (1, 3, 4, 4))
        means = torch.rand(1, 3, 4, 4)
        y = entropy_model.dequantize(x, means)

        assert y.shape == x.shape
        assert y.type() == means.type()

        with pytest.warns(UserWarning):
            yy = entropy_model._dequantize(x, means)
        assert (yy == y).all()

    def test_forward(self, entropy_model):
        with pytest.raises(NotImplementedError):
            entropy_model()

    def test_invalid_coder(self):
        with pytest.raises(ValueError):
            EntropyModel(entropy_coder="huffman")

        with pytest.raises(ValueError):
            EntropyModel(entropy_coder=0xFF)

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
            entropy_model.decompress(["ssss"])

        with pytest.raises(ValueError):
            entropy_model.decompress("sss", torch.rand(1, 3, 4, 4))

        with pytest.raises(ValueError):
            entropy_model.decompress(["sss"], torch.rand(1, 4, 4))

        with pytest.raises(ValueError):
            entropy_model.decompress(["sss"], torch.rand(2, 4, 4))

        with pytest.raises(ValueError):
            entropy_model.decompress(["sss"], torch.rand(1, 4, 4), torch.rand(2, 4, 4))


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

    def test_forward_inference_0D(self):
        entropy_bottleneck = EntropyBottleneck(128)
        entropy_bottleneck.eval()
        x = torch.rand(1, 128)
        y, y_likelihoods = entropy_bottleneck(x)

        assert y.shape == x.shape
        assert y_likelihoods.shape == x.shape

        assert (y == torch.round(x)).all()

    def test_forward_inference_2D(self):
        entropy_bottleneck = EntropyBottleneck(128)
        entropy_bottleneck.eval()
        x = torch.rand(1, 128, 32, 32)
        y, y_likelihoods = entropy_bottleneck(x)

        assert y.shape == x.shape
        assert y_likelihoods.shape == x.shape

        assert (y == torch.round(x)).all()

    def test_forward_inference_ND(self):
        entropy_bottleneck = EntropyBottleneck(128)
        entropy_bottleneck.eval()

        # Test 0D
        x = torch.rand(1, 128)
        y, y_likelihoods = entropy_bottleneck(x)

        assert y.shape == x.shape
        assert y_likelihoods.shape == x.shape

        assert (y == torch.round(x)).all()

        # Test from 1 to 5 dimensions
        for i in range(1, 6):
            x = torch.rand(1, 128, *([4] * i))
            y, y_likelihoods = entropy_bottleneck(x)

            assert y.shape == x.shape
            assert y_likelihoods.shape == x.shape

            assert (y == torch.round(x)).all()

    def test_loss(self):
        entropy_bottleneck = EntropyBottleneck(128)
        loss = entropy_bottleneck.loss()

        assert len(loss.size()) == 0
        assert loss.numel() == 1

    # def test_scripting(self):
    #     entropy_bottleneck = EntropyBottleneck(128)
    #     x = torch.rand(1, 128, 32, 32)

    #     torch.manual_seed(32)
    #     y0 = entropy_bottleneck(x)

    #     m = torch.jit.script(entropy_bottleneck)

    #     torch.manual_seed(32)
    #     y1 = m(x)

    #     assert torch.allclose(y0[0], y1[0])
    #     assert torch.all(y1[1] == 0)  # not yet supported

    def test_update(self):
        # get a pretrained model
        net = bmshj2018_factorized(quality=1, pretrained=True).eval()
        assert not net.update()
        assert not net.update(force=False)
        assert net.update(force=True)

    # def test_script(self):
    #     eb = EntropyBottleneck(32)
    #     eb = torch.jit.script(eb)
    #     x = torch.rand(1, 32, 4, 4)
    #     x_q, likelihoods = eb(x)
    #     assert (likelihoods == torch.zeros_like(x_q)).all()

    def test_compression_2D(self):
        x = torch.rand(1, 128, 32, 32)
        eb = EntropyBottleneck(128)
        eb.update()
        s = eb.compress(x)
        x2 = eb.decompress(s, x.size()[2:])

        assert torch.allclose(torch.round(x), x2)

    def test_compression_ND(self):
        eb = EntropyBottleneck(128)
        eb.update()
        # Test 0D
        x = torch.rand(1, 128)
        s = eb.compress(x)
        x2 = eb.decompress(s, [])

        assert torch.allclose(torch.round(x), x2)

        # Test from 1 to 5 dimensions
        for i in range(1, 6):
            x = torch.rand(1, 128, *([4] * i))
            s = eb.compress(x)
            x2 = eb.decompress(s, x.size()[2:])

            assert torch.allclose(torch.round(x), x2)


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

        with pytest.raises(ValueError):
            GaussianConditional([], scale_bound=-0.1)

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

    # def test_scripting(self):
    #     gaussian_conditional = GaussianConditional(None)
    #     x = torch.rand(1, 128, 32, 32)
    #     scales = torch.rand(1, 128, 32, 32)
    #     means = torch.rand(1, 128, 32, 32)

    #     torch.manual_seed(32)
    #     y0 = gaussian_conditional(x, scales, means)

    #     m = torch.jit.script(gaussian_conditional)

    #     torch.manual_seed(32)
    #     y1 = m(x, scales, means)

    #     assert torch.allclose(y0[0], y1[0])
    #     assert torch.allclose(y0[1], y1[1])

    def test_update(self):
        # get a pretrained model
        net = bmshj2018_hyperprior(quality=1, pretrained=True).eval()
        assert not net.update()
        assert not net.update(force=False)

        quantized_cdf = net.gaussian_conditional._quantized_cdf
        offset = net.gaussian_conditional._offset
        cdf_length = net.gaussian_conditional._cdf_length
        assert net.update(force=True)

        def approx(a, b):
            return ((a - b).abs() <= 2).all()

        assert approx(net.gaussian_conditional._cdf_length, cdf_length)
        assert approx(net.gaussian_conditional._offset, offset)
        assert approx(net.gaussian_conditional._quantized_cdf, quantized_cdf)


@pytest.mark.parametrize(
    "model_cls,args",
    (
        (EntropyBottleneck, (128,)),
        (GaussianConditional, ([0.11, 1.0, 2.0],)),
        (GaussianConditional, (None,)),
    ),
)
def test_deepcopy(model_cls, args):
    model = model_cls(*args)
    model_copy = copy.deepcopy(model)
    x = torch.rand(1, 128, 32, 32)

    if isinstance(model, GaussianConditional):
        opts = (torch.rand_like(x), torch.rand_like(x))
    else:
        opts = ()

    torch.manual_seed(32)
    y0 = model(x, *opts)

    torch.manual_seed(32)
    y1 = model_copy(x, *opts)

    assert torch.allclose(y0[0], y1[0])
