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

from compressai.models._helpers.channel_context import MeanScaleContextHead
from compressai.models._helpers.slice_helpers import (
    infer_max_support_slices,
    infer_num_slices,
    lrp_support_channels,
    make_entropy_transform,
    slice_support_channels,
)


class TestMeanScaleContextHead:
    def test_forward_shape_concatenates_mean_and_scale(self):
        slice_ch, support_ch = 4, 12
        head = MeanScaleContextHead(
            mean_cc=make_entropy_transform(support_ch, slice_ch, widths=(8, 8)),
            scale_cc=make_entropy_transform(support_ch, slice_ch, widths=(8, 8)),
        )
        x = torch.randn(2, support_ch, 4, 4)
        out = head(x)
        assert out.shape == (2, 2 * slice_ch, 4, 4)

    def test_state_dict_paths_split_mean_and_scale(self):
        head = MeanScaleContextHead(
            mean_cc=make_entropy_transform(12, 4, widths=(8, 8)),
            scale_cc=make_entropy_transform(12, 4, widths=(8, 8)),
        )
        keys = set(head.state_dict().keys())
        assert any(k.startswith("mean_cc.") for k in keys)
        assert any(k.startswith("scale_cc.") for k in keys)
        # No support transforms by default -> no associated state.
        assert not any(k.startswith("mean_support_transform.") for k in keys)
        assert not any(k.startswith("scale_support_transform.") for k in keys)

    def test_support_transforms_wrap_inputs(self):
        # Use 1x1 conv that preserves channel count.
        head = MeanScaleContextHead(
            mean_cc=make_entropy_transform(12, 4, widths=(8, 8)),
            scale_cc=make_entropy_transform(12, 4, widths=(8, 8)),
            mean_support_transform=nn.Conv2d(12, 12, 1),
            scale_support_transform=nn.Conv2d(12, 12, 1),
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
        head = MeanScaleContextHead(
            mean_cc=make_entropy_transform(12, 4, widths=(8,)),
            scale_cc=make_entropy_transform(12, 4, widths=(8,)),
            side_split=8,
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

    def test_infer_num_slices_with_y0_context(self):
        # Side-parameter channel-context layout: channel_context covers every
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
        # extra_factor=1 is the default single latent_means concat.
        assert infer_max_support_slices(sd, latent_channels=64, num_slices=8) == 2
