# Copyright (c) 2021-2025, InterDigital Communications, Inc
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

from __future__ import annotations

from typing import Callable, Tuple

import pytest
import torch
import torch.nn as nn

import compressai.layers as layers

from compressai.models._helpers.mlicv2 import (
    ContextReweighting,
    GSCModule,
    HGCPModule,
    RoPE2D,
    SimpleTokenMixing,
    STMAnalysis,
    STMSynthesis,
)


def _round_trip_output(
    make_module: Callable[[], nn.Module],
    inputs: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    torch.manual_seed(0)
    module = make_module().eval()
    with torch.no_grad():
        expected = module(*inputs)

    clone = make_module().eval()
    clone.load_state_dict(module.state_dict())
    with torch.no_grad():
        actual = clone(*inputs)

    assert torch.allclose(actual, expected, atol=1e-6)
    return expected


def test_mlicv2_layers_are_deep_import_only() -> None:
    assert not hasattr(layers, "SimpleTokenMixing")
    assert not hasattr(layers, "HGCPModule")


class TestMlicv2Transforms:
    @staticmethod
    def test_simple_token_mixing_shape_and_state_dict_round_trip() -> None:
        x = torch.randn(2, 8, 8, 8)
        y = _round_trip_output(lambda: SimpleTokenMixing(dim=8), (x,))
        assert y.shape == x.shape

    @staticmethod
    def test_stm_analysis_shape_and_state_dict_round_trip() -> None:
        x = torch.randn(1, 3, 64, 64)
        y = _round_trip_output(lambda: STMAnalysis(N=8, M=16), (x,))
        assert y.shape == (1, 16, 4, 4)

    @staticmethod
    def test_stm_synthesis_shape_and_state_dict_round_trip() -> None:
        y = torch.randn(1, 16, 4, 4)
        x_hat = _round_trip_output(lambda: STMSynthesis(N=8, M=16), (y,))
        assert x_hat.shape == (1, 3, 64, 64)


class TestMlicv2Context:
    @staticmethod
    def test_context_reweighting_shape_attention_and_state_dict_round_trip() -> None:
        x = torch.randn(2, 8, 4, 4)
        y = _round_trip_output(lambda: ContextReweighting(dim=8), (x,))
        assert y.shape == x.shape

        module = ContextReweighting(dim=8).eval()
        with torch.no_grad():
            attention = module.channel_attention(x)
        assert attention.shape == (2, 8, 8)
        assert torch.allclose(
            attention.sum(dim=-1),
            torch.ones(2, 8),
            atol=1e-6,
        )

    @staticmethod
    def test_rope2d_shape_state_dict_and_relative_position_property() -> None:
        module = RoPE2D(dim=4, learnable_thetas=False).eval()
        x = torch.ones(1, 4, 4, 4)
        y = _round_trip_output(lambda: RoPE2D(dim=4, learnable_thetas=False), (x,))
        assert y.shape == x.shape

        rotated = module.rotate(x).reshape(1, 2, 2, 4, 4)
        token_a = rotated[:, :, :, 0, 0].reshape(1, 4)
        token_b = rotated[:, :, :, 1, 1].reshape(1, 4)
        token_c = rotated[:, :, :, 2, 1].reshape(1, 4)
        token_d = rotated[:, :, :, 3, 2].reshape(1, 4)
        score_ab = (token_a * token_b).sum(dim=1)
        score_cd = (token_c * token_d).sum(dim=1)
        assert torch.allclose(score_ab, score_cd, atol=1e-6)

    @staticmethod
    def test_rope2d_rejects_odd_dim() -> None:
        with pytest.raises(ValueError):
            RoPE2D(dim=5)

    @staticmethod
    def test_hgcp_shape_and_state_dict_round_trip() -> None:
        hyper = torch.randn(2, 32, 4, 4)
        y_hat = torch.randn(2, 8, 4, 4)
        y = _round_trip_output(lambda: HGCPModule(M=16, slice_ch=8), (hyper, y_hat))
        assert y.shape == (2, 16, 4, 4)

    @staticmethod
    def test_hgcp_rejects_invalid_head_count() -> None:
        with pytest.raises(ValueError):
            HGCPModule(M=16, slice_ch=7, num_heads=2)

    @staticmethod
    def test_gsc_shape_skip_rate_and_state_dict_round_trip() -> None:
        side_params = torch.randn(2, 12, 4, 4)
        scales = torch.linspace(0.1, 0.5, steps=2 * 8 * 4 * 4).reshape(2, 8, 4, 4)
        means = torch.zeros_like(scales)

        torch.manual_seed(0)
        module = GSCModule(slice_ch=8, side_ch=12, threshold=0.3).eval()
        with torch.no_grad():
            out = module(
                side_params=side_params,
                scales=scales,
                means=means,
                step="anchor",
            )
        selective_map = out["selective_map"]
        assert selective_map.shape == scales.shape
        assert torch.all((selective_map >= 0) & (selective_map <= 1))
        hard_ratio = (selective_map >= 0.5).float().mean().item()
        assert 0.1 < hard_ratio < 0.9

        clone = GSCModule(slice_ch=8, side_ch=12, threshold=0.3).eval()
        clone.load_state_dict(module.state_dict())
        with torch.no_grad():
            cloned = clone(
                side_params=side_params,
                scales=scales,
                means=means,
                step="anchor",
            )
        assert torch.allclose(cloned["selective_map"], selective_map, atol=1e-6)

    @staticmethod
    def test_gsc_rejects_invalid_step() -> None:
        module = GSCModule(slice_ch=4, side_ch=8)
        with pytest.raises(ValueError):
            module(
                side_params=torch.randn(1, 8, 2, 2),
                scales=torch.ones(1, 4, 2, 2),
                means=torch.zeros(1, 4, 2, 2),
                step="bad",
            )
