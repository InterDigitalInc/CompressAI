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
import torch.nn as nn

from compressai.models.priors import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    MeanScaleHyperprior,
    ScaleHyperprior,
    get_scale_table,
)
from compressai.models.utils import (
    _update_registered_buffer,
    find_named_module,
    update_registered_buffers,
)


class TestCompressionModel:
    def test_parameters(self):
        model = CompressionModel(32)
        assert len(list(model.parameters())) == 15
        with pytest.raises(NotImplementedError):
            model(torch.rand(1))

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


class TestModels:
    def test_factorized_prior(self):
        model = FactorizedPrior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2 ** 4
        assert y_likelihoods_shape[3] == x.shape[3] / 2 ** 4

    def test_scale_hyperprior(self, tmpdir):
        model = ScaleHyperprior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2 ** 4
        assert y_likelihoods_shape[3] == x.shape[3] / 2 ** 4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2 ** 6
        assert z_likelihoods_shape[3] == x.shape[3] / 2 ** 6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperprior(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = ScaleHyperprior.from_state_dict(torch.load(filepath))
            assert model.N == loaded.N and model.M == loaded.M

    def test_mean_scale_hyperprior(self):
        model = MeanScaleHyperprior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2 ** 4
        assert y_likelihoods_shape[3] == x.shape[3] / 2 ** 4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2 ** 6
        assert z_likelihoods_shape[3] == x.shape[3] / 2 ** 6

    def test_jarhp(self, tmpdir):
        model = JointAutoregressiveHierarchicalPriors(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2 ** 4
        assert y_likelihoods_shape[3] == x.shape[3] / 2 ** 4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2 ** 6
        assert z_likelihoods_shape[3] == x.shape[3] / 2 ** 6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = JointAutoregressiveHierarchicalPriors(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = JointAutoregressiveHierarchicalPriors.from_state_dict(
                torch.load(filepath)
            )
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
    assert pytest.approx(table[0].item(), 0.02)
    assert pytest.approx(table[-1].item(), 1337)
    assert len(table.size()) == 1
    assert table.size(0) == 32


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 1)


def test_find_named_module():
    assert find_named_module(Foo(), "conv3") is None
    foo = Foo()
    found = find_named_module(foo, "conv1")
    assert found == foo.conv1


def test_update_registered_buffers():
    foo = Foo()
    with pytest.raises(ValueError):
        update_registered_buffers(foo, "conv1", ["qweight"], {})


def test_update_registered_buffer():
    foo = Foo()

    # non-registered buffer
    state_dict = foo.state_dict()
    state_dict["conv1.wweight"] = torch.rand(3)
    with pytest.raises(RuntimeError):
        _update_registered_buffer(
            foo.conv1, "wweight", "conv1.wweight", state_dict, policy="resize"
        )
    with pytest.raises(RuntimeError):
        _update_registered_buffer(
            foo.conv1, "wweight", "conv1.wweight", state_dict, policy="resize_if_empty"
        )
