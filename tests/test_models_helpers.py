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
    GaussianConditionalLatentCodec,
)
from compressai.models._helpers.channel_context import (
    MeanScaleContextHead,
    build_mean_scale_head,
)
from compressai.models._helpers.channel_slice import build_channel_slice_codec


class TestMeanScaleContextHead:
    def test_forward_shape_concatenates_mean_and_scale(self):
        slice_ch, support_ch = 4, 12
        head = build_mean_scale_head(slice_ch, support_ch, widths=(8, 8))
        x = torch.randn(2, support_ch, 4, 4)
        out = head(x)
        assert out.shape == (2, 2 * slice_ch, 4, 4)

    def test_state_dict_paths_split_mean_and_scale(self):
        head = build_mean_scale_head(4, 12, widths=(8, 8))
        keys = set(head.state_dict().keys())
        assert any(k.startswith("mean_cc.") for k in keys)
        assert any(k.startswith("scale_cc.") for k in keys)
        # No support transforms by default -> no associated state.
        assert not any(k.startswith("mean_support_transform.") for k in keys)
        assert not any(k.startswith("scale_support_transform.") for k in keys)

    def test_support_transform_factory_wraps_inputs(self):
        # Use 1x1 conv that preserves channel count.
        def factory(c_in, c_out):
            return nn.Conv2d(c_in, c_out, 1)

        head = build_mean_scale_head(
            4, 12, widths=(8, 8), support_transform_factory=factory
        )
        keys = set(head.state_dict().keys())
        assert any(k.startswith("mean_support_transform.") for k in keys)
        assert any(k.startswith("scale_support_transform.") for k in keys)
        # mean and scale support transforms are independent instances (not shared).
        assert head.mean_support_transform is not head.scale_support_transform

    def test_direct_construction_round_trip(self):
        torch.manual_seed(0)
        mean_cc = nn.Conv2d(12, 4, 1)
        scale_cc = nn.Conv2d(12, 4, 1)
        head = MeanScaleContextHead(mean_cc=mean_cc, scale_cc=scale_cc)
        rebuilt = MeanScaleContextHead(
            mean_cc=nn.Conv2d(12, 4, 1), scale_cc=nn.Conv2d(12, 4, 1)
        )
        rebuilt.load_state_dict(head.state_dict())
        x = torch.randn(2, 12, 4, 4)
        with torch.no_grad():
            assert torch.allclose(head(x), rebuilt(x))

    def test_side_split_routes_means_to_mean_cc_and_scales_to_scale_cc(self):
        # side_split=8 means input is cat(latent_means(8), latent_scales(8), prev_y_hat(4));
        # mean_cc should see cat(latent_means(8), prev_y_hat(4)) = 12 channels;
        # scale_cc same width but reading latent_scales instead of latent_means.
        torch.manual_seed(0)
        head = build_mean_scale_head(
            slice_ch=4, support_ch=20, widths=(8,), side_split=8
        )
        # Sub-network input width = support_ch - side_split = 12.
        first_mean_conv = next(m for m in head.mean_cc if isinstance(m, nn.Conv2d))
        first_scale_conv = next(m for m in head.scale_cc if isinstance(m, nn.Conv2d))
        assert first_mean_conv.in_channels == 12
        assert first_scale_conv.in_channels == 12

        latent_means = torch.randn(2, 8, 4, 4)
        latent_scales = torch.randn(2, 8, 4, 4)
        prev_y_hat = torch.randn(2, 4, 4, 4)
        x = torch.cat([latent_means, latent_scales, prev_y_hat], dim=1)
        with torch.no_grad():
            head_out = head(x)
        assert head_out.shape == (2, 8, 4, 4)
        # Verify routing: mean_cc(cat(latent_means, prev_y_hat)) appears as
        # the second half of head_out (chunks=("scales","means")).
        with torch.no_grad():
            expected_mean = head.mean_cc(torch.cat([latent_means, prev_y_hat], dim=1))
            expected_scale = head.scale_cc(
                torch.cat([latent_scales, prev_y_hat], dim=1)
            )
            scale_out, mean_out = head_out.chunk(2, dim=1)
        assert torch.allclose(scale_out, expected_scale)
        assert torch.allclose(mean_out, expected_mean)


class TestBuildChannelSliceCodec:
    def _leaf_factory(self, side_ch=8):
        # Return a leaf factory whose entropy_parameters width matches what
        # ChannelGroupsLatentCodec hands the leaf at slice k.
        def factory(k, slice_ch):
            if k == 0:
                ctx_in = side_ch
            else:
                ctx_in = (
                    side_ch + 2 * slice_ch
                )  # ch_ctx (= 2*slice_ch from MeanScaleHead) + side
            return GaussianConditionalLatentCodec(
                entropy_parameters=nn.Conv2d(ctx_in, 2 * slice_ch, 1),
            )

        return factory

    def test_dict_keys_y0_through_yK_minus_one(self):
        codec = build_channel_slice_codec(
            groups=[4, 4, 4],
            leaf_factory=self._leaf_factory(side_ch=8),
            channel_context_factory=lambda k, ch, sup: build_mean_scale_head(
                ch, sup, widths=(8, 8)
            ),
        )
        latent_keys = set(codec.latent_codec.keys())
        ctx_keys = set(codec.channel_context.keys())
        assert latent_keys == {"y0", "y1", "y2"}
        # Slice 0 has no channel context entry by design.
        assert ctx_keys == {"y1", "y2"}

    def test_state_dict_paths_match_design_doc(self):
        codec = build_channel_slice_codec(
            groups=[4, 4],
            leaf_factory=self._leaf_factory(side_ch=8),
            channel_context_factory=lambda k, ch, sup: build_mean_scale_head(
                ch, sup, widths=(8,)
            ),
        )
        keys = set(codec.state_dict().keys())
        # Design doc paths (relative to ChannelGroupsLatentCodec root):
        #   channel_context.y{k}.mean_cc.<idx>.weight
        #   channel_context.y{k}.scale_cc.<idx>.weight
        #   latent_codec.y{k}.gaussian_conditional.<buf>
        assert any(k.startswith("channel_context.y1.mean_cc.") for k in keys)
        assert any(k.startswith("channel_context.y1.scale_cc.") for k in keys)
        assert any(k.startswith("latent_codec.y0.") for k in keys)
        assert any(k.startswith("latent_codec.y1.") for k in keys)

    def test_returns_channel_groups_latent_codec(self):
        codec = build_channel_slice_codec(
            groups=[4, 4, 4],
            leaf_factory=self._leaf_factory(side_ch=8),
            channel_context_factory=lambda k, ch, sup: build_mean_scale_head(
                ch, sup, widths=(8, 8)
            ),
        )
        assert isinstance(codec, ChannelGroupsLatentCodec)
        assert codec.groups == [4, 4, 4]
        assert codec.max_support_slices == -1
        assert codec.support_filter is None

    def test_max_support_slices_propagates(self):
        codec = build_channel_slice_codec(
            groups=[4, 4, 4, 4],
            leaf_factory=self._leaf_factory(side_ch=8),
            channel_context_factory=lambda k, ch, sup: build_mean_scale_head(
                ch, sup, widths=(8,)
            ),
            max_support_slices=2,
        )
        assert codec.max_support_slices == 2
        # support_ch passed to channel_context.y3 should be clamped to 2 slices.
        # The MeanScaleContextHead's mean_cc input width is the first conv's
        # in_channels.
        head_y3 = codec.channel_context["y3"]
        first_mean_conv = next(m for m in head_y3.mean_cc if isinstance(m, nn.Conv2d))
        assert first_mean_conv.in_channels == 2 * 4  # 2 slices * 4 ch each

    def test_support_filter_propagates(self):
        def skip_recent(k, prior):
            return prior[: max(k - 1, 0)]

        codec = build_channel_slice_codec(
            groups=[4, 4, 4],
            leaf_factory=self._leaf_factory(side_ch=8),
            channel_context_factory=lambda k, ch, sup: nn.Identity(),
            support_filter=skip_recent,
        )
        assert codec.support_filter is skip_recent

    def test_side_in_context_builds_y0_entry(self):
        # In side_in_context mode the leaf gets only channel_context output
        # (already shaped 2*slice_ch); each leaf can use Identity entropy_parameters.
        codec = build_channel_slice_codec(
            groups=[4, 4, 4],
            side_channels=8,
            side_in_context=True,
            leaf_factory=lambda k, ch: GaussianConditionalLatentCodec(),
            channel_context_factory=lambda k, ch, sup: build_mean_scale_head(
                slice_ch=ch, support_ch=sup, side_split=4, widths=(8,)
            ),
        )
        ctx_keys = set(codec.channel_context.keys())
        assert ctx_keys == {"y0", "y1", "y2"}
        assert codec.side_in_context is True

        # y0 head input width = side_channels = 8.
        head_y0 = codec.channel_context["y0"]
        first_conv_y0 = next(m for m in head_y0.mean_cc if isinstance(m, nn.Conv2d))
        # mean_cc input = support_ch - side_split = 8 - 4 = 4.
        assert first_conv_y0.in_channels == 4

        # y2 head input width = side_channels + groups[0] + groups[1] = 8 + 8 = 16;
        # mean_cc input = 16 - 4 = 12.
        head_y2 = codec.channel_context["y2"]
        first_conv_y2 = next(m for m in head_y2.mean_cc if isinstance(m, nn.Conv2d))
        assert first_conv_y2.in_channels == 12

    def test_side_in_context_requires_side_channels(self):
        with pytest.raises(ValueError, match="side_channels"):
            build_channel_slice_codec(
                groups=[4, 4],
                side_in_context=True,
                leaf_factory=lambda k, ch: GaussianConditionalLatentCodec(),
            )

    def test_side_in_context_forward_runs_end_to_end(self):
        torch.manual_seed(0)
        codec = build_channel_slice_codec(
            groups=[4, 4],
            side_channels=8,
            side_in_context=True,
            leaf_factory=lambda k, ch: GaussianConditionalLatentCodec(),
            channel_context_factory=lambda k, ch, sup: build_mean_scale_head(
                slice_ch=ch, support_ch=sup, side_split=4, widths=(8,)
            ),
        )
        codec.eval()
        y = torch.randn(2, 8, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        with torch.no_grad():
            out = codec(y, side_params)
        assert out["y_hat"].shape == (2, 8, 8, 8)
        assert out["likelihoods"]["y"].shape == (2, 8, 8, 8)
