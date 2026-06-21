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

try:
    from compressai.models._helpers.mlic import (
        AnalysisTransform,
        ChannelContext,
        EntropyParameters,
        HyperAnalysis,
        HyperSynthesis,
        LatentResidualPrediction,
        LinearGlobalInterContext,
        LinearGlobalIntraContext,
        LocalContext,
        StackedCheckerboardConv,
        SynthesisTransform,
        VanillaGlobalInterContext,
        VanillaGlobalIntraContext,
        WindowCheckerboardAttn,
    )
    from compressai.models._helpers.mlic.utils import (
        build_position_index,
        checkerboard_anchor,
        checkerboard_merge,
        checkerboard_nonanchor,
        checkerboard_split,
        squeeze_anchor,
        squeeze_nonanchor,
        unsqueeze_anchor,
        unsqueeze_nonanchor,
    )
except ModuleNotFoundError as err:
    if err.name != "timm":
        raise
    pytestmark = pytest.mark.skip(reason="MLIC++ layers require the [attn] extra")


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


def test_mlic_layers_are_deep_import_only():
    assert not hasattr(layers, "LocalContext")
    assert not hasattr(layers, "EntropyParameters")


class TestMlicContextLayers:
    @staticmethod
    def test_local_context_forward_shape_and_state_dict_round_trip():
        x = torch.randn(2, 4, 4, 4)
        y = _round_trip_output(
            lambda: LocalContext(dim=4, window_size=3, num_heads=2),
            (x,),
        )
        assert y.shape == (2, 8, 4, 4)

    @staticmethod
    def test_context_layers_reject_invalid_head_count():
        with pytest.raises(ValueError):
            LocalContext(dim=5, num_heads=2)
        with pytest.raises(ValueError):
            LinearGlobalInterContext(dim=5, num_heads=2)
        with pytest.raises(ValueError):
            LinearGlobalIntraContext(dim=5, num_heads=2)
        with pytest.raises(ValueError):
            VanillaGlobalInterContext(in_dim=5, num_heads=2)
        with pytest.raises(ValueError):
            VanillaGlobalIntraContext(dim=5, num_heads=2)

    @staticmethod
    def test_channel_context_forward_shape_and_state_dict_round_trip():
        x = torch.randn(2, 8, 4, 4)
        y = _round_trip_output(lambda: ChannelContext(in_dim=8, out_dim=4), (x,))
        assert y.shape == (2, 16, 4, 4)

    @staticmethod
    def test_linear_global_inter_context_shape_and_state_dict_round_trip():
        x = torch.randn(2, 4, 4, 4)
        y = _round_trip_output(
            lambda: LinearGlobalInterContext(dim=4, out_dim=8, num_heads=2),
            (x,),
        )
        assert y.shape == (2, 8, 4, 4)

    @staticmethod
    def test_stacked_checkerboard_conv_shape_and_state_dict_round_trip():
        x = torch.randn(2, 4, 6, 6)
        y = _round_trip_output(
            lambda: StackedCheckerboardConv(dim=4, kernel=5, num_layers=3),
            (x,),
        )
        assert y.shape == (2, 8, 6, 6)

    @staticmethod
    def test_stacked_checkerboard_conv_rejects_even_kernel_or_layers():
        with pytest.raises(ValueError):
            StackedCheckerboardConv(dim=4, kernel=4)
        with pytest.raises(ValueError):
            StackedCheckerboardConv(dim=4, num_layers=2)

    @staticmethod
    def test_window_checkerboard_attention_shape_and_mask():
        x = torch.randn(2, 4, 4, 4)
        y = _round_trip_output(
            lambda: WindowCheckerboardAttn(dim=4, window_size=3, num_heads=2),
            (x,),
        )
        assert y.shape == (2, 8, 4, 4)

        module = WindowCheckerboardAttn(dim=4, window_size=3, num_heads=2)
        module.update_resolution(4, 4, x.device)
        assert module.attn_mask is not None
        assert module.attn_mask.shape == (16, 9, 9)
        assert torch.any(module.attn_mask == 0)
        assert torch.any(module.attn_mask == -100)
        assert torch.all((module.attn_mask == 0) | (module.attn_mask == -100))

    @staticmethod
    def test_vanilla_global_inter_context_shape_and_state_dict_round_trip():
        x = torch.randn(2, 4, 4, 4)
        y = _round_trip_output(
            lambda: VanillaGlobalInterContext(in_dim=4, out_dim=8, num_heads=2),
            (x,),
        )
        assert y.shape == (2, 8, 4, 4)

    @staticmethod
    def test_vanilla_global_intra_context_shape_mask_and_state_dict_round_trip():
        x1 = torch.randn(2, 4, 4, 4)
        x2 = torch.randn(2, 4, 4, 4)
        y = _round_trip_output(
            lambda: VanillaGlobalIntraContext(dim=4, num_heads=2),
            (x1, x2),
        )
        assert y.shape == (2, 8, 4, 4)

        module = VanillaGlobalIntraContext(dim=4, num_heads=2, local_mask_radius=0)
        mask = module._attention_mask(4, 4, x1.device)
        assert mask.shape == (8, 8)
        assert not torch.any(mask)

        module = VanillaGlobalIntraContext(dim=4, num_heads=2, local_mask_radius=1)
        mask = module._attention_mask(4, 4, x1.device)
        assert mask.shape == (8, 8)
        assert torch.any(mask)
        assert not torch.all(mask)

    @staticmethod
    def test_linear_global_intra_context_shape_and_state_dict_round_trip():
        x1 = torch.randn(2, 4, 4, 4)
        x2 = torch.randn(2, 4, 4, 4)
        y = _round_trip_output(
            lambda: LinearGlobalIntraContext(dim=4, num_heads=2),
            (x1, x2),
        )
        assert y.shape == (2, 8, 4, 4)


class TestMlicTransforms:
    @staticmethod
    def test_analysis_transform_shape_and_state_dict_round_trip():
        x = torch.randn(1, 3, 64, 64)
        module = AnalysisTransform(N=8, M=16)
        assert isinstance(module.analysis_transform[0].act, nn.GELU)
        assert not hasattr(module.analysis_transform[0], "leaky_relu")
        y = _round_trip_output(lambda: module, (x,))
        assert y.shape == (1, 16, 4, 4)

    @staticmethod
    def test_synthesis_transform_shape_and_state_dict_round_trip():
        y = torch.randn(1, 16, 4, 4)
        module = SynthesisTransform(N=8, M=16)
        assert isinstance(module.synthesis_transform[0].act, nn.GELU)
        assert not hasattr(module.synthesis_transform[0], "leaky_relu")
        x_hat = _round_trip_output(lambda: module, (y,))
        assert x_hat.shape == (1, 3, 64, 64)

    @staticmethod
    def test_hyper_analysis_shape_and_state_dict_round_trip():
        y = torch.randn(1, 16, 4, 4)
        z = _round_trip_output(lambda: HyperAnalysis(M=16, N=8), (y,))
        assert z.shape == (1, 8, 1, 1)

    @staticmethod
    def test_hyper_synthesis_shape_and_state_dict_round_trip():
        z = torch.randn(1, 8, 1, 1)
        params = _round_trip_output(lambda: HyperSynthesis(M=16, N=8), (z,))
        assert params.shape == (1, 32, 4, 4)

    @staticmethod
    def test_entropy_parameters_shape_and_state_dict_round_trip():
        params = torch.randn(2, 10, 4, 4)
        y = _round_trip_output(
            lambda: EntropyParameters(in_dim=10, out_dim=8), (params,)
        )
        assert y.shape == (2, 8, 4, 4)

    @staticmethod
    def test_lrp_shape_bound_and_state_dict_round_trip():
        params = torch.randn(2, 10, 4, 4)
        y = _round_trip_output(
            lambda: LatentResidualPrediction(in_dim=10, out_dim=4),
            (params,),
        )
        assert y.shape == (2, 4, 4, 4)
        assert torch.all(y <= 0.5)
        assert torch.all(y >= -0.5)


class TestMlicCheckerboardUtils:
    @staticmethod
    def test_checkerboard_split_merge_and_squeeze_layout():
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        anchor, nonanchor = checkerboard_split(x)

        expected_anchor_squeezed = torch.tensor(
            [[[[1.0, 3.0], [4.0, 6.0], [9.0, 11.0], [12.0, 14.0]]]]
        )
        expected_nonanchor_squeezed = torch.tensor(
            [[[[0.0, 2.0], [5.0, 7.0], [8.0, 10.0], [13.0, 15.0]]]]
        )

        assert torch.equal(checkerboard_merge(anchor, nonanchor), x)
        assert torch.equal(checkerboard_anchor(x), anchor)
        assert torch.equal(checkerboard_nonanchor(x), nonanchor)
        assert torch.equal(squeeze_anchor(x), expected_anchor_squeezed)
        assert torch.equal(squeeze_nonanchor(x), expected_nonanchor_squeezed)
        assert torch.equal(unsqueeze_anchor(squeeze_anchor(x)), anchor)
        assert torch.equal(unsqueeze_nonanchor(squeeze_nonanchor(x)), nonanchor)

    @staticmethod
    def test_build_position_index_shape_and_center_value():
        position_index = build_position_index((3, 3))

        assert position_index.shape == (9, 9)
        assert position_index[4, 4].item() == 12
        assert position_index.min().item() == 0
        assert position_index.max().item() == 24
