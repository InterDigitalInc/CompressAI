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
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.models.google import (
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
from compressai.models.vbr import ScaleHyperpriorVbr
from compressai.models.video.google import ScaleSpaceFlow


class DummyCompressionModel(CompressionModel):
    def __init__(self, entropy_bottleneck_channels):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)


class TestCompressionModel:
    def test_parameters(self):
        model = DummyCompressionModel(32)
        assert len(list(model.parameters())) == 15
        with pytest.raises(NotImplementedError):
            model(torch.rand(1))

    def test_init(self):
        class Model(DummyCompressionModel):
            def __init__(self):
                super().__init__(3)
                self.conv = nn.Conv2d(3, 3, 3)
                self.deconv = nn.ConvTranspose2d(3, 3, 3)
                self.original_conv = self.conv.weight
                self.original_deconv = self.deconv.weight

        model = Model()
        nn.init.kaiming_normal_(model.original_conv)
        nn.init.kaiming_normal_(model.original_deconv)

        assert torch.allclose(model.original_conv, model.conv.weight)
        assert torch.allclose(model.original_deconv, model.deconv.weight)


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
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

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
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperprior(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = ScaleHyperprior.from_state_dict(torch.load(filepath))
            assert model.N == loaded.N and model.M == loaded.M

    def test_scale_hyperprior_vbr(self, tmpdir):
        model = ScaleHyperpriorVbr(128, 192, vr_entbttlnck=True)
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
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperpriorVbr(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = ScaleHyperpriorVbr.from_state_dict(torch.load(filepath))
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
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

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
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = JointAutoregressiveHierarchicalPriors(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = JointAutoregressiveHierarchicalPriors.from_state_dict(
                torch.load(filepath)
            )
            assert model.N == loaded.N and model.M == loaded.M

    def test_scale_space_flow(self):
        model = ScaleSpaceFlow()
        x = [torch.rand(1, 3, 128, 128), torch.rand(1, 3, 128, 128)]
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "keyframe" in out["likelihoods"][0]
        assert "y" in out["likelihoods"][0]["keyframe"]
        assert "z" in out["likelihoods"][0]["keyframe"]

        assert "motion" in out["likelihoods"][1]
        assert "y" in out["likelihoods"][1]["motion"]
        assert "z" in out["likelihoods"][1]["motion"]

        assert "residual" in out["likelihoods"][1]
        assert "y" in out["likelihoods"][1]["residual"]
        assert "z" in out["likelihoods"][1]["residual"]

        assert out["x_hat"][0].shape == x[0].shape
        assert out["x_hat"][1].shape == x[1].shape

        y_likelihoods_shape = out["likelihoods"][0]["keyframe"]["y"].shape
        assert y_likelihoods_shape[0] == x[0].shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x[0].shape[2] / 2**4
        assert y_likelihoods_shape[3] == x[0].shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"][0]["keyframe"]["z"].shape
        assert z_likelihoods_shape[0] == x[0].shape[0]
        assert z_likelihoods_shape[1] == 192
        assert z_likelihoods_shape[2] == x[0].shape[2] / 2**7  # (128x128 input)
        assert z_likelihoods_shape[3] == x[0].shape[3] / 2**7

        y_likelihoods_shape = out["likelihoods"][1]["motion"]["y"].shape
        assert y_likelihoods_shape[0] == x[1].shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x[1].shape[2] / 2**4
        assert y_likelihoods_shape[3] == x[1].shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"][1]["motion"]["z"].shape
        assert z_likelihoods_shape[0] == x[1].shape[0]
        assert z_likelihoods_shape[1] == 192
        assert z_likelihoods_shape[2] == x[1].shape[2] / 2**7  # (128x128 input)
        assert z_likelihoods_shape[3] == x[1].shape[3] / 2**7

        y_likelihoods_shape = out["likelihoods"][1]["residual"]["y"].shape
        assert y_likelihoods_shape[0] == x[1].shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x[1].shape[2] / 2**4
        assert y_likelihoods_shape[3] == x[1].shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"][1]["residual"]["z"].shape
        assert z_likelihoods_shape[0] == x[1].shape[0]
        assert z_likelihoods_shape[1] == 192
        assert z_likelihoods_shape[2] == x[1].shape[2] / 2**7  # (128x128 input)
        assert z_likelihoods_shape[3] == x[1].shape[3] / 2**7


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
    assert pytest.approx(table[0].item()) == 0.02
    assert pytest.approx(table[-1].item()) == 1337
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
