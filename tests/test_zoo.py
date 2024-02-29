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
    bmshj2018_factorized_relu,
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
            net = bmshj2018_factorized(i, metric="mse", progress=False)
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(i, metric="mse", progress=False)
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
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, metric):
        for i in range(1, 6):
            net = bmshj2018_factorized(
                i, metric=metric, pretrained=True, progress=False
            )
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(
                i, metric=metric, pretrained=True, progress=False
            )
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestBmshj2018FactorizedReLU:
    def test_params(self):
        for i in range(1, 6):
            net = bmshj2018_factorized_relu(i, metric="mse", progress=False)
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized_relu(i, metric="mse")
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            bmshj2018_factorized_relu(-1)

        with pytest.raises(ValueError):
            bmshj2018_factorized_relu(10)

        with pytest.raises(ValueError):
            bmshj2018_factorized_relu(10, metric="ssim")

        with pytest.raises(ValueError):
            bmshj2018_factorized_relu(1, metric="ssim")


class TestBmshj2018Hyperprior:
    def test_params(self):
        for i in range(1, 6):
            net = bmshj2018_hyperprior(i, metric="mse", progress=False)
            assert isinstance(net, ScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_hyperprior(i, metric="mse", progress=False)
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
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, metric):
        # test we can load the correct models from the urls
        for i in range(1, 6):
            net = bmshj2018_hyperprior(
                i, metric=metric, pretrained=True, progress=False
            )
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_hyperprior(
                i, metric=metric, pretrained=True, progress=False
            )
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestMbt2018Mean:
    def test_parameters(self):
        for i in range(1, 5):
            net = mbt2018_mean(i, metric="mse", progress=False)
            assert isinstance(net, MeanScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018_mean(i, metric="mse", progress=False)
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
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, metric):
        # test we can load the correct models from the urls
        for i in range(1, 5):
            net = mbt2018_mean(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018_mean(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestMbt2018:
    def test_ok(self):
        for i in range(1, 5):
            net = mbt2018(i, metric="mse", progress=False)
            assert isinstance(net, JointAutoregressiveHierarchicalPriors)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018(i, metric="mse", progress=False)
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
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, metric):
        # test we can load the correct models from the urls
        for i in range(1, 5):
            net = mbt2018(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018(i, metric=metric, pretrained=True, progress=False)
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
            net = func(i, metric="mse", progress=False)
            assert isinstance(net, cls)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 128

        for i in range(4, 7):
            net = func(i, metric="mse", progress=False)
            assert isinstance(net, cls)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 192

    @pytest.mark.slow
    @pytest.mark.pretrained
    @pytest.mark.parametrize("model_entrypoint", (cheng2020_anchor, cheng2020_attn))
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, model_entrypoint, metric):
        for i in range(1, 4):
            net = model_entrypoint(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 128

        for i in range(4, 7):
            net = model_entrypoint(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) in (128, 192)
