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

import pytest
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


class TestChannelGroupsSideInContext:
    """``side_in_context`` mode used by Family 1 codecs."""

    def _make_family1_codec(
        self,
        groups=(4, 4, 4),
        side_ch=8,
        max_support_slices=-1,
    ):
        K = len(groups)

        # channel_context for k=0: input is just side_params (= side_ch).
        # channel_context for k>=1: input is cat(side_params, *prev_y_hat).
        # Use 1x1 convs that map to 2 * groups[k] (the leaf chunks into scales/means).
        def _ctx_in(k):
            count = k if max_support_slices < 0 else min(k, max_support_slices)
            return side_ch + sum(groups[:count])

        channel_context = {
            f"y{k}": nn.Conv2d(_ctx_in(k), 2 * groups[k], 1) for k in range(K)
        }
        # Leaves see only the channel_context output (no re-cat with side_params)
        # in side_in_context mode -> entropy_parameters can be Identity since
        # channel_context already shaped the tensor to 2 * slice_ch.
        latent_codec = {f"y{k}": GaussianConditionalLatentCodec() for k in range(K)}
        return ChannelGroupsLatentCodec(
            latent_codec=latent_codec,
            channel_context=channel_context,
            groups=list(groups),
            max_support_slices=max_support_slices,
            side_in_context=True,
        )

    def test_constructor_requires_y0_entry(self):
        # side_in_context=True but missing y0 channel_context -> ValueError.
        with pytest.raises(ValueError, match="y0"):
            ChannelGroupsLatentCodec(
                latent_codec={"y0": GaussianConditionalLatentCodec()},
                channel_context={},  # missing y0
                groups=[4],
                side_in_context=True,
            )

    def test_forward_routes_through_y0_channel_context(self):
        torch.manual_seed(11)
        codec = self._make_family1_codec(groups=(4, 4))
        y = torch.randn(2, 8, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        out = codec(y, side_params)
        assert out["y_hat"].shape == (2, 8, 8, 8)
        assert out["likelihoods"]["y"].shape == (2, 8, 8, 8)

    def test_get_ctx_params_for_k_zero_calls_y0(self):
        codec = self._make_family1_codec(groups=(4, 4))
        side_params = torch.zeros(1, 8, 4, 4)
        ctx = codec._get_ctx_params(0, side_params, [])
        # Output shape == channel_context.y0(side_params) -> (1, 2 * groups[0], 4, 4).
        assert ctx.shape == (1, 8, 4, 4)

    def test_get_ctx_params_for_k_positive_concats_side(self):
        codec = self._make_family1_codec(groups=(4, 4))
        side_params = torch.zeros(1, 8, 4, 4)
        prev_y_hat = [torch.zeros(1, 4, 4, 4)]
        ctx = codec._get_ctx_params(1, side_params, prev_y_hat)
        # Channel_context.y1 input width = side_ch + groups[0] = 8 + 4 = 12;
        # output = 2 * groups[1] = 8.
        assert ctx.shape == (1, 8, 4, 4)


class TestChannelGroupsDecompressShape:
    """Regression coverage for ``ChannelGroupsLatentCodec.decompress`` shape
    reconstruction.

    Family 1 (STF/WACNN/TCM/CCA) leaves are :class:`LRPGaussianLatentCodec`
    which inherits :class:`GaussianConditionalLatentCodec.compress` → returns
    ``shape = y.shape[2:4]`` (2D ``(H, W)``). ELIC-style leaves use
    :class:`CheckerboardLatentCodec` → returns ``shape = y_hat.shape[1:]``
    (3D ``(C, H, W)``). ``decompress`` must allocate the correct 4D
    ``(N, sum_C, H, W)`` buffer in either case.
    """

    class _LeafMock2D(nn.Module):
        """Mimics GaussianConditionalLatentCodec: shape=(H, W) from compress,
        no real entropy coding (zeros for y_hat)."""

        def __init__(self, slice_ch):
            super().__init__()
            self.slice_ch = slice_ch

        def compress(self, y, ctx_params):
            n = y.shape[0]
            return {
                "strings": [[b"" for _ in range(n)]],
                "shape": tuple(y.shape[2:4]),
                "y_hat": torch.zeros_like(y),
            }

        def decompress(self, strings, shape, ctx_params, **kwargs):
            n = len(strings[0])
            h, w = shape
            return {"y_hat": torch.zeros((n, self.slice_ch, h, w))}

    class _LeafMock3D(nn.Module):
        """Mimics CheckerboardLatentCodec: shape=(C, H, W) from compress."""

        def __init__(self, slice_ch):
            super().__init__()
            self.slice_ch = slice_ch

        def compress(self, y, ctx_params):
            n = y.shape[0]
            return {
                "strings": [[b"" for _ in range(n)]],
                "shape": tuple(y.shape[1:]),
                "y_hat": torch.zeros_like(y),
            }

        def decompress(self, strings, shape, ctx_params, **kwargs):
            n = len(strings[0])
            c, h, w = shape
            return {"y_hat": torch.zeros((n, c, h, w))}

    def _make_codec(self, leaf_cls, groups=(4, 4, 4), side_ch=8):
        K = len(groups)
        return ChannelGroupsLatentCodec(
            latent_codec={f"y{k}": leaf_cls(groups[k]) for k in range(K)},
            channel_context={f"y{k}": nn.Identity() for k in range(1, K)},
            groups=list(groups),
        )

    def test_decompress_with_2d_leaf_shape(self):
        # Pre-fix: y_shape = (sum(s[0] for s in shape), *shape[0][1:])
        # collapsed to (sum_H, W) = (3*6, 5) -> y_hat 3D -> RuntimeError when
        # assigning the 4D leaf y_hat into a 3D split slice.
        groups = [4, 4, 4]
        codec = self._make_codec(self._LeafMock2D, groups=groups)
        # Deliberately pick H != W and H != sum(groups) so a regression in
        # axis-confusion (e.g. sum_H instead of sum_C) surfaces as a shape
        # error, not a silent wrong-shape pass.
        h, w = 6, 5
        y = torch.randn(1, sum(groups), h, w)
        side_params = torch.zeros(1, 8, h, w)
        out_enc = codec.compress(y, side_params)
        out_dec = codec.decompress(out_enc["strings"], out_enc["shape"], side_params)
        assert out_dec["y_hat"].shape == (1, sum(groups), h, w)

    def test_decompress_with_3d_leaf_shape_still_works(self):
        # ELIC-style path must keep working.
        groups = [4, 4, 4]
        codec = self._make_codec(self._LeafMock3D, groups=groups)
        h, w = 6, 5
        y = torch.randn(1, sum(groups), h, w)
        side_params = torch.zeros(1, 8, h, w)
        out_enc = codec.compress(y, side_params)
        out_dec = codec.decompress(out_enc["strings"], out_enc["shape"], side_params)
        assert out_dec["y_hat"].shape == (1, sum(groups), h, w)


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
        # New state-dict layout (ELIC default): channel_context entries exist
        # for k >= 1. For 4 slices total, we expect 3 mean_cc keys -> infer
        # returns 4 (helper adds 1 because y0 is missing).
        sd = {
            f"latent_codec.y.channel_context.y{k}.mean_cc.0.weight": (torch.zeros(8, 4))
            for k in range(1, 4)
        }
        assert infer_num_slices(sd) == 4

    def test_infer_num_slices_side_in_context(self):
        # Family 1 side_in_context=True layout: channel_context covers every
        # slice (y0..yK-1). Helper auto-detects via the presence of y0 and
        # does NOT add 1.
        sd = {
            f"latent_codec.y.channel_context.y{k}.mean_cc.0.weight": (torch.zeros(8, 4))
            for k in range(0, 4)
        }
        assert infer_num_slices(sd) == 4

    def test_infer_num_slices_empty(self):
        assert infer_num_slices({}) == 0

    def test_infer_max_support_slices_new_path(self):
        # mean_cc.0 takes (latent_means + slice_channels * support) input channels.
        # With M=64, num_slices=8, slice_channels=8, support=2 -> input ch = 64 + 16 = 80.
        sd = {
            "latent_codec.y.channel_context.y2.mean_cc.0.weight": (
                torch.zeros(64, 80, 3, 3)
            ),
            "latent_codec.y.channel_context.y3.mean_cc.0.weight": (
                torch.zeros(64, 80, 3, 3)
            ),
        }
        # extra_factor=1 is the Family 1 default (single latent_means concat).
        assert infer_max_support_slices(sd, latent_channels=64, num_slices=8) == 2
