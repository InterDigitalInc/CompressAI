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


class TestChannelGroupsLatentCodecExtensions:
    def _make_codec(
        self,
        groups=(4, 4, 4),
        side_ch=8,
        support_slices=None,
    ):
        K = len(groups)
        effective_support = (
            [list(range(k)) for k in range(K)]
            if support_slices is None
            else support_slices
        )
        channel_context = {f"y{k}": nn.Identity() for k in range(1, K)}

        def _ctx_in(k):
            if k == 0:
                return side_ch
            return side_ch + sum(groups[j] for j in effective_support[k])

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
            support_slices=support_slices,
        )

    def test_default_support_slices_uses_all_prior(self):
        codec = self._make_codec()
        assert codec.support_slices == [(), (0,), (0, 1)]

    def test_explicit_support_slices_are_preserved(self):
        support_slices = [[], [0], [0], [0, 2]]
        codec = self._make_codec(
            groups=(4, 4, 4, 4),
            support_slices=support_slices,
        )
        assert codec.support_slices == [(), (0,), (0,), (0, 2)]

    def test_support_slices_reject_current_or_future_groups(self):
        with pytest.raises(AssertionError):
            self._make_codec(groups=(4, 4), support_slices=[[], [1]])

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

    def test_explicit_support_slices_runs_forward(self):
        torch.manual_seed(3)
        codec = self._make_codec(
            groups=(4, 4, 4, 4),
            support_slices=[[], [0], [0], [0, 2]],
        )
        y = torch.randn(2, 16, 8, 8)
        side_params = torch.randn(2, 8, 8, 8)
        out = codec(y, side_params)
        assert out["y_hat"].shape == (2, 16, 8, 8)


class TestChannelGroupsDecompressShape:
    """Coverage for the unified channel-first latent-codec shape convention."""

    class _LeafMock(nn.Module):
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
            assert c == self.slice_ch
            return {"y_hat": torch.zeros((n, c, h, w))}

    def _make_codec(self, groups=(4, 4, 4)):
        K = len(groups)
        return ChannelGroupsLatentCodec(
            latent_codec={f"y{k}": self._LeafMock(groups[k]) for k in range(K)},
            channel_context={f"y{k}": nn.Identity() for k in range(1, K)},
            groups=list(groups),
        )

    def test_decompress_passes_per_group_channel_shape(self):
        groups = [4, 4, 4]
        codec = self._make_codec(groups=groups)
        h, w = 6, 5
        y = torch.randn(1, sum(groups), h, w)
        side_params = torch.zeros(1, 8, h, w)
        out_enc = codec.compress(y, side_params)
        assert out_enc["shape"] == y.shape[1:]
        out_dec = codec.decompress(out_enc["strings"], out_enc["shape"], side_params)
        assert out_dec["y_hat"].shape == (1, sum(groups), h, w)
