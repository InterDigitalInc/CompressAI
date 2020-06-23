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
import torch.nn as nn

import pytest

from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior,
                            mbt2018_mean, mbt2018, cheng2020_anchor,
                            cheng2020_attn)
from compressai.zoo.image import _load_model

from compressai.models.priors import (SCALES_LEVELS, SCALES_MAX, SCALES_MIN,
                                      CompressionModel, FactorizedPrior,
                                      MeanScaleHyperprior, ScaleHyperprior,
                                      JointAutoregressiveHierarchicalPriors,
                                      get_scale_table)
from compressai.models import Cheng2020Anchor, Cheng2020Attention

from compressai.models.utils import (find_named_module,
                                     update_registered_buffers,
                                     _update_registered_buffer)


class TestCompressionModel:
    def test_parameters(self):
        model = CompressionModel(32)
        assert len(list(model.parameters())) == 0

        with pytest.raises(NotImplementedError):
            model(torch.rand(1))

    def test_aux_parameters(self):
        model = CompressionModel(32)
        for m in model.aux_parameters():
            assert m.shape[0] == 32  # channel-based

    def test_init(self):
        class Model(CompressionModel):
            def __init__(self):
                super().__init__(3)
                self.conv = nn.Conv2d(3, 3, 3)
                self.deconv = nn.ConvTranspose2d(3, 3, 3)
                self.original_conv = self.conv.weight
                self.original_deconv = self.deconv.weight

                self._initialize_weights()

        model = Model()
        nn.init.kaiming_normal_(model.original_conv)
        nn.init.kaiming_normal_(model.original_deconv)

        assert torch.allclose(model.original_conv, model.conv.weight)
        assert torch.allclose(model.original_deconv, model.deconv.weight)

        assert model.conv.bias.abs().sum() == 0
        assert model.deconv.bias.abs().sum() == 0


class TestLoadModel:
    def test_invalid(self):
        with pytest.raises(ValueError):
            _load_model('yolo', 'mse', 1)

        with pytest.raises(ValueError):
            _load_model('mbt2018', 'mse', 0)


class TestModels:
    def test_factorized_prior(self):
        model = FactorizedPrior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert 'x_hat' in out
        assert 'likelihoods' in out
        assert 'y' in out['likelihoods']

        assert out['x_hat'].shape == x.shape

        y_likelihoods_shape = out['likelihoods']['y'].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

    def test_scale_hyperprior(self, tmpdir):
        model = ScaleHyperprior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert 'x_hat' in out
        assert 'likelihoods' in out
        assert 'y' in out['likelihoods']
        assert 'z' in out['likelihoods']

        assert out['x_hat'].shape == x.shape

        y_likelihoods_shape = out['likelihoods']['y'].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out['likelihoods']['z'].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperprior(*sz)
            filepath = tmpdir.join('model.pth.rar').strpath
            torch.save(model.state_dict(), filepath)
            loaded = ScaleHyperprior.from_state_dict(torch.load(filepath))
            assert model.N == loaded.N and model.M == loaded.M

    def test_mean_scale_hyperprior(self):
        model = MeanScaleHyperprior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert 'x_hat' in out
        assert 'likelihoods' in out
        assert 'y' in out['likelihoods']
        assert 'z' in out['likelihoods']

        assert out['x_hat'].shape == x.shape

        y_likelihoods_shape = out['likelihoods']['y'].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out['likelihoods']['z'].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

    def test_jarhp(self, tmpdir):
        model = JointAutoregressiveHierarchicalPriors(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert 'x_hat' in out
        assert 'likelihoods' in out
        assert 'y' in out['likelihoods']
        assert 'z' in out['likelihoods']

        assert out['x_hat'].shape == x.shape

        y_likelihoods_shape = out['likelihoods']['y'].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out['likelihoods']['z'].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = JointAutoregressiveHierarchicalPriors(*sz)
            filepath = tmpdir.join('model.pth.rar').strpath
            torch.save(model.state_dict(), filepath)
            loaded = JointAutoregressiveHierarchicalPriors.from_state_dict(
                torch.load(filepath))
            assert model.N == loaded.N and model.M == loaded.M


def test_scale_table_default():
    table = get_scale_table()
    assert SCALES_MIN == 0.11
    assert SCALES_MAX == 256
    assert SCALES_LEVELS == 64
    assert table[0] == SCALES_MIN
    assert table[-1] == SCALES_MAX
    assert len(table.size()) == 1
    assert table.size(0) == SCALES_LEVELS


def test_scale_table_custom():
    table = get_scale_table(0.02, 1337, 32)
    assert table[0] == 0.02
    assert table[-1] == 1337
    assert len(table.size()) == 1
    assert table.size(0) == 32


class TestBmshj2018Factorized:
    def test_params(self):
        for i in range(1, 6):
            net = bmshj2018_factorized(i, metric='mse')
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()['g_a.0.weight'].size(0) == 128
            assert net.state_dict()['g_a.6.weight'].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(i, metric='mse')
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            bmshj2018_factorized(-1)

        with pytest.raises(ValueError):
            bmshj2018_factorized(10)

        with pytest.raises(ValueError):
            bmshj2018_factorized(10, metric='ssim')

        with pytest.raises(ValueError):
            bmshj2018_factorized(1, metric='ssim')

    def test_pretrained(self):
        # test we can load the correct models from the urls
        for i in range(1, 6):
            net = bmshj2018_factorized(i, metric='mse', pretrained=True)
            assert net.state_dict()['g_a.0.weight'].size(0) == 128
            assert net.state_dict()['g_a.6.weight'].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(i, metric='mse', pretrained=True)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192
            assert net.state_dict()['g_a.6.weight'].size(0) == 320


class TestBmshj2018Hyperprior:
    def test_params(self):
        for i in range(1, 6):
            net = bmshj2018_hyperprior(i, metric='mse')
            assert isinstance(net, ScaleHyperprior)
            assert net.state_dict()['g_a.0.weight'].size(0) == 128
            assert net.state_dict()['g_a.6.weight'].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_hyperprior(i, metric='mse')
            assert isinstance(net, ScaleHyperprior)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192
            assert net.state_dict()['g_a.6.weight'].size(0) == 320

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            bmshj2018_hyperprior(-1)

        with pytest.raises(ValueError):
            bmshj2018_hyperprior(10)

        with pytest.raises(ValueError):
            bmshj2018_hyperprior(10, metric='ssim')

        with pytest.raises(ValueError):
            bmshj2018_hyperprior(1, metric='ssim')

    def test_pretrained(self):
        # test we can load the correct models from the urls
        for i in range(1, 6):
            net = bmshj2018_factorized(i, metric='mse', pretrained=True)
            assert net.state_dict()['g_a.0.weight'].size(0) == 128
            assert net.state_dict()['g_a.6.weight'].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(i, metric='mse', pretrained=True)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192
            assert net.state_dict()['g_a.6.weight'].size(0) == 320


class TestMbt2018Mean:
    def test_parameters(self):
        for i in range(1, 5):
            net = mbt2018_mean(i, metric='mse')
            assert isinstance(net, MeanScaleHyperprior)
            assert net.state_dict()['g_a.0.weight'].size(0) == 128
            assert net.state_dict()['g_a.6.weight'].size(0) == 192

        for i in range(5, 9):
            net = mbt2018_mean(i, metric='mse')
            assert isinstance(net, MeanScaleHyperprior)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192
            assert net.state_dict()['g_a.6.weight'].size(0) == 320

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            mbt2018_mean(-1)

        with pytest.raises(ValueError):
            mbt2018_mean(10)

        with pytest.raises(ValueError):
            mbt2018_mean(10, metric='ssim')

        with pytest.raises(ValueError):
            mbt2018_mean(1, metric='ssim')

    def test_pretrained(self):
        # test we can load the correct models from the urls
        for i in range(1, 5):
            net = mbt2018_mean(i, metric='mse', pretrained=True)
            assert net.state_dict()['g_a.0.weight'].size(0) == 128
            assert net.state_dict()['g_a.6.weight'].size(0) == 192

        for i in range(5, 9):
            net = mbt2018_mean(i, metric='mse', pretrained=True)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192
            assert net.state_dict()['g_a.6.weight'].size(0) == 320


class TestMbt2018:
    def test_ok(self):
        for i in range(1, 5):
            net = mbt2018(i, metric='mse')
            assert isinstance(net, JointAutoregressiveHierarchicalPriors)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192
            assert net.state_dict()['g_a.6.weight'].size(0) == 192

        for i in range(5, 9):
            net = mbt2018(i, metric='mse')
            assert isinstance(net, JointAutoregressiveHierarchicalPriors)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192
            assert net.state_dict()['g_a.6.weight'].size(0) == 320

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            mbt2018(-1)

        with pytest.raises(ValueError):
            mbt2018(10)

        with pytest.raises(ValueError):
            mbt2018(10, metric='ssim')

        with pytest.raises(ValueError):
            mbt2018(1, metric='ssim')

    def test_pretrained(self):
        # test we can load the correct models from the urls
        for i in range(1, 5):
            net = mbt2018(i, metric='mse', pretrained=True)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192
            assert net.state_dict()['g_a.6.weight'].size(0) == 192

        for i in range(5, 9):
            net = mbt2018(i, metric='mse', pretrained=True)
            assert net.state_dict()['g_a.0.weight'].size(0) == 192
            assert net.state_dict()['g_a.6.weight'].size(0) == 320


class TestCheng2020:
    @pytest.mark.parametrize('func,cls', (
        (cheng2020_anchor, Cheng2020Anchor),
        (cheng2020_attn, Cheng2020Attention),
    ))
    def test_anchor_ok(self, func, cls):
        for i in range(1, 4):
            net = func(i, metric='mse')
            assert isinstance(net, cls)
            assert net.state_dict()['g_a.0.conv1.weight'].size(0) == 128

        for i in range(4, 7):
            net = func(i, metric='mse')
            assert isinstance(net, cls)
            assert net.state_dict()['g_a.0.conv1.weight'].size(0) == 192


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 1)


def test_find_named_module():
    assert find_named_module(Foo(), 'conv3') is None
    foo = Foo()
    found = find_named_module(foo, 'conv1')
    assert found == foo.conv1


def test_update_registered_buffers():
    foo = Foo()
    with pytest.raises(ValueError):
        update_registered_buffers(foo, 'conv1', ['qweight'], {})


def test_update_registered_buffer():
    foo = Foo()

    # non-registered buffer
    state_dict = foo.state_dict()
    state_dict['conv1.wweight'] = torch.rand(3)
    with pytest.raises(RuntimeError):
        _update_registered_buffer(foo.conv1,
                                  'wweight',
                                  'conv1.wweight',
                                  state_dict,
                                  policy='resize')
    with pytest.raises(RuntimeError):
        _update_registered_buffer(foo.conv1,
                                  'wweight',
                                  'conv1.wweight',
                                  state_dict,
                                  policy='resize_if_empty')
