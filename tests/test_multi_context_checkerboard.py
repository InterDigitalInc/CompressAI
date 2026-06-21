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

import torch
import torch.nn as nn

from compressai.latent_codecs import (
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    MultiContextCheckerboardLatentCodec,
)
from compressai.layers import CheckerboardMaskedConv2d


class _ZeroContext(nn.Module):
    """Stand-in spatial context that emits zeros with a fixed channel count.

    Used by the ELIC-equivalence regression: upstream ``CheckerboardLatentCodec``
    feeds the anchor pass an all-zero tensor sized to ``context_prediction.out``;
    ``MultiContextCheckerboardLatentCodec`` skips the spatial slot entirely when
    ``spatial_context_anchor=None``, so the regression has to supply this zero
    context explicitly.
    """

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = int(out_channels)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y.new_zeros(y.shape[0], self.out_channels, y.shape[2], y.shape[3])


class _ZeroEntropyParameters(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = int(out_channels)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return params.new_zeros(
            params.shape[0],
            self.out_channels,
            params.shape[2],
            params.shape[3],
        )


class _ConstantResidual(nn.Module):
    def __init__(self, out_channels: int, value: float) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.value = float(value)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return params.new_full(
            (params.shape[0], self.out_channels, params.shape[2], params.shape[3]),
            self.value,
        )


class _ConvSelectivePredictor(nn.Module):
    def __init__(self, side_channels: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(side_channels + 2 * channels, channels, 1)

    def forward(
        self,
        *,
        side_params: torch.Tensor,
        scales: torch.Tensor,
        means: torch.Tensor,
        step: str,
    ) -> torch.Tensor:
        return torch.sigmoid(self.proj(torch.cat([side_params, scales, means], dim=1)))


class TestMultiContextCheckerboardLatentCodec:
    class _IntraContext(nn.Module):
        def __init__(self, side_ch=8, y_ch=4, out_ch=3):
            super().__init__()
            self.proj = nn.Conv2d(side_ch + y_ch, out_ch, 1)

        def forward(self, side_params, anchor_y_hat):
            return self.proj(torch.cat([side_params, anchor_y_hat], dim=1))

    @staticmethod
    def _scale_table():
        return [0.11, 0.5, 1.0, 2.0, 4.0]

    def _make(
        self,
        *,
        y_ch=4,
        side_ch=8,
        anchor_in=None,
        nonanchor_in=None,
        **kwargs,
    ):
        """Construct a codec with caller-controlled head input widths.

        ``anchor_in`` / ``nonanchor_in`` default to ``side_ch`` because the
        codec now omits any ``spatial_context_*=None`` slot from the
        entropy-parameters input. Tests that supply spatial / intra-channel
        hooks must widen the corresponding head.
        """
        if anchor_in is None:
            anchor_in = side_ch
        if nonanchor_in is None:
            nonanchor_in = side_ch
        return MultiContextCheckerboardLatentCodec(
            entropy_parameters_anchor=nn.Conv2d(anchor_in, 2 * y_ch, 1),
            entropy_parameters_nonanchor=nn.Conv2d(nonanchor_in, 2 * y_ch, 1),
            scale_table=self._scale_table(),
            **kwargs,
        )

    def test_default_forward_shapes(self):
        codec = self._make().eval()
        y = torch.randn(2, 4, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        with torch.no_grad():
            out = codec(y, side_params)
        assert out["y_hat"].shape == (2, 4, 8, 8)
        assert out["likelihoods"]["y"].shape == (2, 4, 8, 8)

    def test_anchor_skips_spatial_context_when_none(self):
        """``spatial_context_anchor=None`` must NOT pad with zeros.

        Anchor head sized to ``side_ch`` only — would crash with channel
        mismatch if the leaf still emitted a ``y_ch``-wide zero block.
        Locks in the skip semantics required for MLIC++ k=0 anchor wiring.
        """
        codec = self._make(
            anchor_in=8,  # side_ch only
            nonanchor_in=8 + 4,  # side_ch + spatial_context_nonanchor.out
            spatial_context_nonanchor=CheckerboardMaskedConv2d(4, 4, 5, padding=2),
        ).eval()
        y = torch.randn(2, 4, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        with torch.no_grad():
            out = codec(y, side_params)
        assert out["y_hat"].shape == y.shape

    def test_spatial_nonanchor_forward_shapes(self):
        codec = self._make(
            nonanchor_in=8 + 4,  # side_ch + spatial_context_nonanchor.out
            spatial_context_nonanchor=CheckerboardMaskedConv2d(4, 4, 5, padding=2),
        ).eval()
        y = torch.randn(2, 4, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        with torch.no_grad():
            out = codec(y, side_params)
        assert out["y_hat"].shape == y.shape
        assert out["likelihoods"]["y"].shape == y.shape

    def test_all_hooks_forward_shapes_and_state_dict_paths(self):
        def lrp_inputs(side_params, params, y_hat):
            return torch.cat([side_params, params, y_hat], dim=1)

        codec = MultiContextCheckerboardLatentCodec(
            entropy_parameters_anchor=nn.Conv2d(4 + 8, 8, 1),
            entropy_parameters_nonanchor=nn.Conv2d(4 + 8 + 3, 8, 1),
            scale_table=self._scale_table(),
            spatial_context_anchor=nn.Conv2d(4, 4, 1),
            spatial_context_nonanchor=CheckerboardMaskedConv2d(4, 4, 5, padding=2),
            intra_channel_context_nonanchor=self._IntraContext(),
            lrp_anchor=nn.Conv2d(8 + 8 + 4, 4, 1),
            lrp_nonanchor=nn.Conv2d(8 + 8 + 4, 4, 1),
            lrp_input_builder=lrp_inputs,
            selective_predictor=_ConvSelectivePredictor(8, 4),
        ).eval()
        keys = set(codec.state_dict().keys())
        assert any(k.startswith("entropy_parameters_anchor.") for k in keys)
        assert any(k.startswith("entropy_parameters_nonanchor.") for k in keys)
        assert any(k.startswith("spatial_context_anchor.") for k in keys)
        assert any(k.startswith("spatial_context_nonanchor.") for k in keys)
        assert any(k.startswith("intra_channel_context_nonanchor.") for k in keys)
        assert any(k.startswith("selective_predictor.") for k in keys)
        assert any(k.startswith("lrp_anchor.") for k in keys)
        assert any(k.startswith("lrp_nonanchor.") for k in keys)
        assert any(k.startswith("y.gaussian_conditional.") for k in keys)

        y = torch.randn(2, 4, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        with torch.no_grad():
            out = codec(y, side_params)
        assert out["y_hat"].shape == y.shape

    def test_lrp_activation_can_be_skipped(self):
        codec = MultiContextCheckerboardLatentCodec(
            entropy_parameters_anchor=_ZeroEntropyParameters(8),
            entropy_parameters_nonanchor=_ZeroEntropyParameters(8),
            scale_table=self._scale_table(),
            lrp_anchor=_ConstantResidual(4, 0.25),
            lrp_nonanchor=_ConstantResidual(4, 0.25),
            lrp_activation=None,
            lrp_scale=1.0,
        ).eval()
        y = torch.zeros(1, 4, 4, 4)
        side_params = torch.zeros(1, 8, 4, 4)

        with torch.no_grad():
            out = codec(y, side_params)

        assert torch.allclose(out["y_hat"], torch.full_like(y, 0.25))

    def test_state_dict_round_trip(self):
        torch.manual_seed(13)
        kwargs = dict(
            nonanchor_in=8 + 4,
            spatial_context_nonanchor=CheckerboardMaskedConv2d(4, 4, 5, padding=2),
        )
        codec = self._make(**kwargs).eval()
        reconstructed = self._make(**kwargs).eval()
        reconstructed.load_state_dict(codec.state_dict())
        y = torch.randn(2, 4, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        with torch.no_grad():
            out_a = codec(y, side_params)
            out_b = reconstructed(y, side_params)
        assert torch.allclose(out_a["y_hat"], out_b["y_hat"])
        assert torch.allclose(out_a["likelihoods"]["y"], out_b["likelihoods"]["y"])

    def test_compress_decompress_round_trip(self):
        torch.manual_seed(17)
        codec = self._make(
            nonanchor_in=8 + 4,
            spatial_context_nonanchor=CheckerboardMaskedConv2d(4, 4, 5, padding=2),
        ).eval()
        codec.y.gaussian_conditional.update()
        y = torch.randn(1, 4, 8, 8)
        side_params = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            forward = codec(y, side_params)
        compressed = codec.compress(y, side_params)
        decompressed = codec.decompress(
            compressed["strings"], compressed["shape"], side_params
        )
        assert torch.allclose(forward["y_hat"], compressed["y_hat"])
        assert torch.allclose(compressed["y_hat"], decompressed["y_hat"])

    def test_matches_checkerboard_latent_codec_when_heads_are_shared(self):
        torch.manual_seed(19)
        y_ch, side_ch = 4, 8
        # Upstream CheckerboardLatentCodec: shared head + masked-conv spatial
        # context. Anchor pass feeds an all-zero tensor of width
        # ``context_prediction.out_channels`` to the head.
        entropy_parameters = nn.Conv2d(y_ch + side_ch, 2 * y_ch, 1)
        context_prediction = CheckerboardMaskedConv2d(y_ch, y_ch, 5, padding=2)
        base = CheckerboardLatentCodec(
            latent_codec={
                "y": GaussianConditionalLatentCodec(scale_table=self._scale_table())
            },
            entropy_parameters=entropy_parameters,
            context_prediction=context_prediction,
        ).eval()
        # Sibling leaf must reproduce that anchor-pass zero slot explicitly,
        # because spatial_context_anchor=None is now "skip the slot" not
        # "pad with y-shaped zeros". Wider regression for arbitrary
        # context_prediction.out_channels != y_ch is enabled by this design.
        generalized = MultiContextCheckerboardLatentCodec(
            entropy_parameters_anchor=entropy_parameters,
            entropy_parameters_nonanchor=entropy_parameters,
            spatial_context_anchor=_ZeroContext(context_prediction.out_channels),
            spatial_context_nonanchor=context_prediction,
            latent_codec={
                "y": GaussianConditionalLatentCodec(scale_table=self._scale_table())
            },
        ).eval()
        y = torch.randn(2, y_ch, 8, 8)
        side_params = torch.randn(2, side_ch, 8, 8)
        with torch.no_grad():
            base_out = base(y, side_params)
            generalized_out = generalized(y, side_params)
        assert torch.allclose(base_out["y_hat"], generalized_out["y_hat"])
        assert torch.allclose(
            base_out["likelihoods"]["y"], generalized_out["likelihoods"]["y"]
        )
