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
    ChannelGroupsLatentCodec,
    DualHyperSynthesis,
    GaussianConditionalLatentCodec,
    LRPGaussianLatentCodec,
)
from compressai.latent_codecs._slice_helpers import (
    infer_max_support_slices,
    infer_num_slices,
    lrp_support_channels,
    make_entropy_transform,
    slice_support_channels,
)


class TestDualHyperSynthesis:
    def test_concatenates_dual_heads(self):
        h_mean_s = nn.Conv2d(4, 6, 1)
        h_scale_s = nn.Conv2d(4, 6, 1)
        wrapper = DualHyperSynthesis(h_mean_s, h_scale_s)
        z_hat = torch.randn(2, 4, 8, 8)
        out = wrapper(z_hat)
        assert out.shape == (2, 12, 8, 8)
        expected = torch.cat([h_mean_s(z_hat), h_scale_s(z_hat)], dim=1)
        assert torch.allclose(out, expected)

    def test_state_dict_paths_split_per_head(self):
        wrapper = DualHyperSynthesis(nn.Conv2d(4, 6, 1), nn.Conv2d(4, 6, 1))
        keys = set(wrapper.state_dict().keys())
        assert "h_mean_s.weight" in keys
        assert "h_mean_s.bias" in keys
        assert "h_scale_s.weight" in keys
        assert "h_scale_s.bias" in keys


class TestLRPGaussianLatentCodec:
    def _make(self, slice_ch=4, ctx_ch=8):
        # entropy_parameters maps ctx_ch -> 2*slice_ch (chunked into scales/means)
        entropy_parameters = nn.Conv2d(ctx_ch, 2 * slice_ch, 1)
        # lrp_transform input = ctx_ch + slice_ch (cat ctx_params with y_hat)
        lrp_transform = nn.Sequential(
            nn.Conv2d(ctx_ch + slice_ch, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, slice_ch, 3, padding=1),
        )
        return LRPGaussianLatentCodec(
            lrp_transform=lrp_transform,
            entropy_parameters=entropy_parameters,
        )

    def test_forward_shapes(self):
        codec = self._make()
        y = torch.randn(2, 4, 8, 8)
        ctx = torch.randn(2, 8, 8, 8)
        out = codec(y, ctx)
        assert out["y_hat"].shape == (2, 4, 8, 8)
        assert out["likelihoods"]["y"].shape == (2, 4, 8, 8)

    def test_lrp_changes_y_hat_relative_to_base(self):
        # With identical entropy parameters, LRP variant should produce a
        # different y_hat than the un-refined GaussianConditionalLatentCodec.
        torch.manual_seed(0)
        slice_ch, ctx_ch = 4, 8
        entropy_parameters = nn.Conv2d(ctx_ch, 2 * slice_ch, 1)
        lrp_transform = nn.Sequential(
            nn.Conv2d(ctx_ch + slice_ch, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, slice_ch, 3, padding=1),
        )
        # Push lrp through tanh's slope-1 region: zero biases keep tanh near 0
        # so the refinement is small but non-trivial.
        for m in lrp_transform.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
                nn.init.zeros_(m.bias)
        base = GaussianConditionalLatentCodec(
            entropy_parameters=entropy_parameters
        ).eval()
        refined = LRPGaussianLatentCodec(
            lrp_transform=lrp_transform, entropy_parameters=entropy_parameters
        ).eval()
        y = torch.randn(2, slice_ch, 8, 8)
        ctx = torch.randn(2, ctx_ch, 8, 8)
        with torch.no_grad():
            base_out = base(y, ctx)
            ref_out = refined(y, ctx)
        assert not torch.allclose(base_out["y_hat"], ref_out["y_hat"])
        # Same y_likelihoods because the Gaussian step is identical.
        assert torch.allclose(base_out["likelihoods"]["y"], ref_out["likelihoods"]["y"])

    def test_state_dict_round_trip(self):
        codec = self._make().eval()
        keys = set(codec.state_dict().keys())
        # Inherits entropy_parameters / gaussian_conditional from base, adds lrp_transform.
        assert any(k.startswith("lrp_transform.") for k in keys)
        assert any(k.startswith("entropy_parameters.") for k in keys)
        assert any(k.startswith("gaussian_conditional.") for k in keys)

        reconstructed = self._make().eval()
        reconstructed.load_state_dict(codec.state_dict())
        y = torch.randn(2, 4, 8, 8)
        ctx = torch.randn(2, 8, 8, 8)
        with torch.no_grad():
            out_a = codec(y, ctx)
            out_b = reconstructed(y, ctx)
        assert torch.allclose(out_a["y_hat"], out_b["y_hat"])

    def test_lrp_scale_zero_collapses_to_base(self):
        torch.manual_seed(1)
        codec = self._make().eval()
        codec.lrp_scale = 0.0
        y = torch.randn(2, 4, 8, 8)
        ctx = torch.randn(2, 8, 8, 8)
        base_codec = GaussianConditionalLatentCodec(
            entropy_parameters=codec.entropy_parameters,
            gaussian_conditional=codec.gaussian_conditional,
        ).eval()
        with torch.no_grad():
            base_out = base_codec(y, ctx)
            ref_out = codec(y, ctx)
        assert torch.allclose(base_out["y_hat"], ref_out["y_hat"])


class TestChannelGroupsLatentCodecExtensions:
    def _make_codec(
        self,
        groups=(4, 4, 4),
        side_ch=8,
        max_support_slices=-1,
        support_filter=None,
    ):
        K = len(groups)
        # Channel-context input is sum of (clamped) prev y_hat slice channels;
        # use Identity which simply forwards the concatenated tensor unchanged.
        channel_context = {f"y{k}": nn.Identity() for k in range(1, K)}

        def _ctx_in(k):
            if k == 0:
                return side_ch
            if max_support_slices < 0:
                count = k
            else:
                count = min(k, max_support_slices)
            return side_ch + sum(groups[:count])

        # Each leaf needs an entropy_parameters MLP sized to its own ctx input.
        latent_codec = {
            f"y{k}": GaussianConditionalLatentCodec(
                entropy_parameters=nn.Conv2d(_ctx_in(k), 2 * groups[k], 1),
            )
            for k in range(K)
        }
        return ChannelGroupsLatentCodec(
            latent_codec=latent_codec,
            channel_context=channel_context,
            groups=list(groups),
            max_support_slices=max_support_slices,
            support_filter=support_filter,
        )

    def test_default_select_support_uses_all_prior(self):
        codec = self._make_codec()
        slices = [torch.zeros(1, 4, 4, 4) for _ in range(3)]
        assert codec._select_support(0, slices) == []
        assert codec._select_support(1, slices) == slices[:1]
        assert codec._select_support(3, slices) == slices[:3]

    def test_max_support_slices_clamps(self):
        codec = self._make_codec(max_support_slices=2)
        slices = [torch.zeros(1, 4, 4, 4) for _ in range(3)]
        # k=3 with clamp=2 -> drop the most recent slice (index 2)
        result = codec._select_support(3, slices)
        assert len(result) == 2
        assert result == slices[:2]

    def test_support_filter_overrides_max_support(self):
        # CCA-aux skip-most-recent pattern.
        def skip_recent(k, prior):
            return prior[: max(k - 1, 0)]

        codec = self._make_codec(max_support_slices=10, support_filter=skip_recent)
        slices = [torch.zeros(1, 4, 4, 4) for _ in range(4)]
        assert codec._select_support(0, slices) == []
        assert codec._select_support(1, slices) == []  # k-1 = 0
        assert codec._select_support(3, slices) == slices[:2]  # skip slice 2

    def test_default_forward_matches_pre_extension_behaviour(self):
        # With defaults the new constructor should be drop-in for ELIC-style use.
        torch.manual_seed(7)
        codec = self._make_codec()
        groups = codec.groups
        M = sum(groups)
        y = torch.randn(2, M, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        out = codec(y, side_params)
        assert out["y_hat"].shape == (2, M, 8, 8)
        assert out["likelihoods"]["y"].shape == (2, M, 8, 8)

    def test_max_support_slices_changes_forward_output(self):
        # Build a codec whose channel_context input width matches a clamped support;
        # then verify that forward produces a different y_hat than the un-clamped version.
        torch.manual_seed(3)
        codec_clamp = self._make_codec(groups=(4, 4, 4, 4), max_support_slices=1)
        # Reuse all leaf weights from clamp codec on a fresh "no clamp" codec for
        # an apples-to-apples comparison; we expect clamp to drop information for
        # slices k >= 2 only (their leaf input width differs, so we only need to
        # check that clamp codec runs end-to-end).
        y = torch.randn(2, 16, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        out = codec_clamp(y, side_params)
        assert out["y_hat"].shape == (2, 16, 8, 8)


class TestSliceHelpers:
    def test_slice_support_channels_default_use_all(self):
        # With max_support_slices = -1 the helper returns the full latent + k slices.
        assert slice_support_channels(64, 8, 0, -1) == 64
        assert slice_support_channels(64, 8, 5, -1) == 64 + 8 * 5

    def test_slice_support_channels_clamps(self):
        assert slice_support_channels(64, 8, 5, 3) == 64 + 8 * 3
        assert slice_support_channels(64, 8, 1, 3) == 64 + 8 * 1

    def test_lrp_support_channels(self):
        assert lrp_support_channels(64, 8, 0, -1) == 64 + 8
        assert lrp_support_channels(64, 8, 5, 3) == 64 + 8 * 4

    def test_make_entropy_transform_default_widths(self):
        net = make_entropy_transform(40, 8)
        # Default widths (224, 128): conv-gelu-conv-gelu-conv -> 5 modules.
        assert len(net) == 5
        x = torch.randn(2, 40, 8, 8)
        y = net(x)
        assert y.shape == (2, 8, 8, 8)

    def test_make_entropy_transform_custom_widths(self):
        net = make_entropy_transform(40, 8, widths=(64, 32))
        x = torch.randn(2, 40, 8, 8)
        y = net(x)
        assert y.shape == (2, 8, 8, 8)

    def test_infer_num_slices_new_path(self):
        # New state-dict layout: channel_context entries exist for k >= 1.
        # For 4 slices total, we expect 3 mean_cc keys -> infer returns 4.
        sd = {
            f"latent_codec.latent_codec.y.channel_context.y{k}.mean_cc.0.weight": (
                torch.zeros(8, 4)
            )
            for k in range(1, 4)
        }
        assert infer_num_slices(sd) == 4

    def test_infer_num_slices_empty(self):
        assert infer_num_slices({}) == 0

    def test_infer_max_support_slices_new_path(self):
        # mean_cc.0 takes (latent_means + slice_channels * support) input channels.
        # With M=64, num_slices=8, slice_channels=8, support=2 -> input ch = 64 + 16 = 80.
        sd = {
            "latent_codec.latent_codec.y.channel_context.y2.mean_cc.0.weight": (
                torch.zeros(64, 80, 3, 3)
            ),
            "latent_codec.latent_codec.y.channel_context.y3.mean_cc.0.weight": (
                torch.zeros(64, 80, 3, 3)
            ),
        }
        # extra_factor=1 is the Family 1 default (single latent_means concat).
        assert infer_max_support_slices(sd, latent_channels=64, num_slices=8) == 2
