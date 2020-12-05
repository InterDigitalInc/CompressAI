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

from compressai.models import (
    Cheng2020Anchor,
    Cheng2020Attention,
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    MeanScaleHyperprior,
    ScaleHyperprior,
)
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    cheng2020_attn,
    mbt2018,
    mbt2018_mean,
)
from compressai.zoo.image import _load_model


class TestLoadModel:
    def test_invalid(self):
        with pytest.raises(ValueError):
            _load_model("yolo", "mse", 1)

        with pytest.raises(ValueError):
            _load_model("mbt2018", "mse", 0)


class TestBmshj2018Factorized:
    def test_params(self):
        for i in range(1, 6):
            net = bmshj2018_factorized(i, metric="mse")
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(i, metric="mse")
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            bmshj2018_factorized(-1)

        with pytest.raises(ValueError):
            bmshj2018_factorized(10)

        with pytest.raises(ValueError):
            bmshj2018_factorized(10, metric="ssim")

        with pytest.raises(ValueError):
            bmshj2018_factorized(1, metric="ssim")

    @pytest.mark.slow
    @pytest.mark.pretrained
    @pytest.mark.parametrize(
        "metric", [("mse",), ("ms-ssim",)]
    )  # bypass weird pytest bug
    def test_pretrained(self, metric):
        metric = metric[0]
        for i in range(1, 6):
            net = bmshj2018_factorized(i, metric=metric, pretrained=True)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(i, metric=metric, pretrained=True)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestBmshj2018Hyperprior:
    def test_params(self):
        for i in range(1, 6):
            net = bmshj2018_hyperprior(i, metric="mse")
            assert isinstance(net, ScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_hyperprior(i, metric="mse")
            assert isinstance(net, ScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            bmshj2018_hyperprior(-1)

        with pytest.raises(ValueError):
            bmshj2018_hyperprior(10)

        with pytest.raises(ValueError):
            bmshj2018_hyperprior(10, metric="ssim")

        with pytest.raises(ValueError):
            bmshj2018_hyperprior(1, metric="ssim")

    @pytest.mark.slow
    @pytest.mark.pretrained
    def test_pretrained(self):
        # test we can load the correct models from the urls
        for i in range(1, 6):
            net = bmshj2018_factorized(i, metric="mse", pretrained=True)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(i, metric="mse", pretrained=True)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestMbt2018Mean:
    def test_parameters(self):
        for i in range(1, 5):
            net = mbt2018_mean(i, metric="mse")
            assert isinstance(net, MeanScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018_mean(i, metric="mse")
            assert isinstance(net, MeanScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            mbt2018_mean(-1)

        with pytest.raises(ValueError):
            mbt2018_mean(10)

        with pytest.raises(ValueError):
            mbt2018_mean(10, metric="ssim")

        with pytest.raises(ValueError):
            mbt2018_mean(1, metric="ssim")

    @pytest.mark.slow
    @pytest.mark.pretrained
    def test_pretrained(self):
        # test we can load the correct models from the urls
        for i in range(1, 5):
            net = mbt2018_mean(i, metric="mse", pretrained=True)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018_mean(i, metric="mse", pretrained=True)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestMbt2018:
    def test_ok(self):
        for i in range(1, 5):
            net = mbt2018(i, metric="mse")
            assert isinstance(net, JointAutoregressiveHierarchicalPriors)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018(i, metric="mse")
            assert isinstance(net, JointAutoregressiveHierarchicalPriors)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            mbt2018(-1)

        with pytest.raises(ValueError):
            mbt2018(10)

        with pytest.raises(ValueError):
            mbt2018(10, metric="ssim")

        with pytest.raises(ValueError):
            mbt2018(1, metric="ssim")

    @pytest.mark.slow
    @pytest.mark.pretrained
    def test_pretrained(self):
        # test we can load the correct models from the urls
        for i in range(1, 5):
            net = mbt2018(i, metric="mse", pretrained=True)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018(i, metric="mse", pretrained=True)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestCheng2020:
    @pytest.mark.parametrize(
        "func,cls",
        (
            (cheng2020_anchor, Cheng2020Anchor),
            (cheng2020_attn, Cheng2020Attention),
        ),
    )
    def test_anchor_ok(self, func, cls):
        for i in range(1, 4):
            net = func(i, metric="mse")
            assert isinstance(net, cls)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 128

        for i in range(4, 7):
            net = func(i, metric="mse")
            assert isinstance(net, cls)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 192

    @pytest.mark.slow
    @pytest.mark.pretrained
    def test_pretrained(self):
        for i in range(1, 4):
            net = cheng2020_anchor(i, metric="mse", pretrained=True)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 128

        for i in range(4, 7):
            net = cheng2020_anchor(i, metric="mse", pretrained=True)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 192
