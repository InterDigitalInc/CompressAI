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


class TestSharedDictionary:
    def test_dt_shape_and_state_dict_path(self):
        from compressai.models._helpers.dictionary_context import SharedDictionary

        shared = SharedDictionary(dict_num=16, dictionary_dim=64)
        assert shared.dt.shape == (16, 64)
        assert list(shared.state_dict().keys()) == ["dt"]

    def test_expand_for_broadcasts_without_copy(self):
        from compressai.models._helpers.dictionary_context import SharedDictionary

        shared = SharedDictionary(dict_num=8, dictionary_dim=32)
        out = shared.expand_for(4)
        assert out.shape == (4, 8, 32)
        # All B copies share storage with the underlying dt
        assert out.data_ptr() == shared.dt.data_ptr()


class TestBuildDictionaryMeanScaleHead:
    def _build(self, *, emit_mean_support=False):
        from compressai.models._helpers.dictionary_context import (
            SharedDictionary,
            build_dictionary_mean_scale_head,
        )

        # Tiny config: M=32, slice_ch=8, support_count=2
        m = 32
        slice_ch = 8
        support_count = 2
        support_ch = 2 * m + slice_ch * support_count
        shared = SharedDictionary(dict_num=8, dictionary_dim=64)
        head = build_dictionary_mean_scale_head(
            slice_ch=slice_ch,
            support_ch=support_ch,
            shared_dictionary=shared,
            dict_output_ch=m,
            cross_attention_kwargs={"head_num": 4, "mlp_rate": 2},
            widths=(16,),
            emit_mean_support=emit_mean_support,
        )
        return shared, head, m, slice_ch, support_ch

    def test_forward_shape_no_emit(self):
        shared, head, m, slice_ch, support_ch = self._build(emit_mean_support=False)
        x = torch.randn(2, support_ch, 4, 4)
        out = head(x)
        # Output: cat([scale, mean]) → 2 * slice_ch
        assert out.shape == (2, 2 * slice_ch, 4, 4)

    def test_forward_shape_with_emit_mean_support(self):
        shared, head, m, slice_ch, support_ch = self._build(emit_mean_support=True)
        x = torch.randn(2, support_ch, 4, 4)
        out = head(x)
        # Output: cat([scale, mean, support]) where support = cat([x, dict_info(M)])
        expected = 2 * slice_ch + (support_ch + m)
        assert out.shape == (2, expected, 4, 4)

    def test_dt_not_duplicated_in_head_state_dict(self):
        shared, head, *_ = self._build()
        head_keys = list(head.state_dict().keys())
        assert all(
            "dt" not in k for k in head_keys
        ), f"dt leaked into head.state_dict: {[k for k in head_keys if 'dt' in k]}"

    def test_dt_appears_once_in_container_state_dict(self):
        from compressai.models._helpers.dictionary_context import (
            SharedDictionary,
            build_dictionary_mean_scale_head,
        )

        m, slice_ch, support_count = 32, 8, 2
        support_ch = 2 * m + slice_ch * support_count

        class _Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_dictionary = SharedDictionary(dict_num=8, dictionary_dim=64)
                self.heads = nn.ModuleDict(
                    {
                        f"y{k}": build_dictionary_mean_scale_head(
                            slice_ch=slice_ch,
                            support_ch=support_ch,
                            shared_dictionary=self.shared_dictionary,
                            dict_output_ch=m,
                            cross_attention_kwargs={"head_num": 4, "mlp_rate": 2},
                            widths=(16,),
                        )
                        for k in range(3)
                    }
                )

        container = _Container()
        dt_keys = [k for k in container.state_dict() if k.endswith(".dt")]
        assert dt_keys == [
            "shared_dictionary.dt"
        ], f"expected single shared_dictionary.dt path, got: {dt_keys}"


class TestOLP:
    @staticmethod
    def test_forward_shape_square():
        from compressai.models._helpers.auxt import OLP

        m = OLP(8, 8)
        out = m(torch.randn(2, 8))
        assert out.shape == (2, 8)

    @staticmethod
    def test_loss_returns_scalar_for_each_aspect_ratio():
        from compressai.models._helpers.auxt import OLP

        for in_dim, out_dim in [(8, 8), (16, 4), (4, 16)]:
            m = OLP(in_dim, out_dim)
            loss = m.loss()
            assert loss.dim() == 0, f"OLP({in_dim}, {out_dim}).loss() must be scalar"
            assert torch.isfinite(loss)

    @staticmethod
    def test_state_dict_round_trip():
        from compressai.models._helpers.auxt import OLP

        m = OLP(8, 8)
        m2 = OLP(8, 8)
        m2.load_state_dict(m.state_dict(), strict=True)
        x = torch.randn(2, 8)
        assert torch.allclose(m(x), m2(x))


class TestWLSiWLS:
    @staticmethod
    def test_wls_iwls_shapes_and_round_trip():
        pytest.importorskip("pytorch_wavelets")
        from compressai.models._helpers.auxt import WLS, iWLS

        wls = WLS(in_dim=3, out_dim=8)
        iwls = iWLS(in_dim=8, out_dim=3)
        x = torch.randn(2, 3, 16, 16)
        y = wls(x)
        # WLS halves spatial size (DWT) and produces out_dim channels.
        assert y.shape == (2, 8, 8, 8)
        z = iwls(y)
        assert z.shape == x.shape

        # state_dict round-trip on WLS.
        wls2 = WLS(in_dim=3, out_dim=8)
        wls2.load_state_dict(wls.state_dict(), strict=True)
        assert torch.allclose(wls(x), wls2(x))

    @staticmethod
    def test_aux_loss_returns_zero_when_no_olp_present():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import aux_loss

        # A toy model with no OLP submodules — aux_loss should return a 0-d
        # zero Tensor so callers can unconditionally add it to the objective.
        model = _nn.Sequential(_nn.Linear(8, 8))
        loss = aux_loss(model)
        assert loss.dim() == 0
        assert loss.item() == 0.0

    @staticmethod
    def test_aux_loss_aggregates_olp_modules():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import OLP, aux_loss

        # Two OLPs at different positions in the tree — aux_loss should
        # equal the sum of their individual losses.
        class _Container(_nn.Module):
            def __init__(self):
                super().__init__()
                self.a = OLP(8, 8)
                self.b = OLP(16, 4)

        c = _Container()
        expected = c.a.loss() + c.b.loss()
        assert torch.allclose(aux_loss(c), expected)


class TestForwardWithAuxt:
    @staticmethod
    def test_collapses_to_transform_when_aux_layers_none():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import forward_with_auxt

        transform = _nn.Sequential(_nn.Conv2d(3, 4, 1), _nn.Conv2d(4, 5, 1))
        x = torch.randn(2, 3, 4, 4)
        with torch.no_grad():
            assert torch.allclose(
                forward_with_auxt(transform, None, (), x), transform(x)
            )

    @staticmethod
    def test_sums_auxt_at_merge_positions():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import forward_with_auxt

        # transform: 4 layers, all identity Conv2d-style (1x1, weight=I).
        def _identity_conv(ch):
            conv = _nn.Conv2d(ch, ch, 1, bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.eye(ch).view(ch, ch, 1, 1))
            return conv

        transform = _nn.Sequential(*(_identity_conv(3) for _ in range(4)))
        # AuxT branch with 2 layers, also identity. Merge at position 1 and 3.
        aux = _nn.ModuleList([_identity_conv(3), _identity_conv(3)])
        x = torch.randn(1, 3, 2, 2)
        out = forward_with_auxt(transform, aux, (1, 3), x)
        # After identity transform + 2 identity AuxT additions, output = 3 * x
        # (x at start + AuxT[0]=x at pos 1 + AuxT[1]=AuxT[0]=x at pos 3).
        assert torch.allclose(out, 3 * x)

    @staticmethod
    def test_raises_when_merge_positions_underrun_aux_depth():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import forward_with_auxt

        transform = _nn.Sequential(_nn.Conv2d(3, 3, 1), _nn.Conv2d(3, 3, 1))
        aux = _nn.ModuleList([_nn.Conv2d(3, 3, 1), _nn.Conv2d(3, 3, 1)])
        x = torch.randn(1, 3, 2, 2)
        # Only 1 merge position for 2 aux layers -> mismatch.
        with pytest.raises(RuntimeError, match="merge positions"):
            forward_with_auxt(transform, aux, (0,), x)


class TestAuxtStateDictHelpers:
    @staticmethod
    def test_has_auxt_state():
        from compressai.models._helpers.auxt import has_auxt_state

        assert has_auxt_state({"AuxT_enc.0.olp.linear.weight": torch.zeros(2)})
        assert has_auxt_state({"AuxT_dec.3.scaling_factors": torch.zeros(2)})
        assert not has_auxt_state({"g_a.0.weight": torch.zeros(2)})

    @staticmethod
    def test_is_auxt_wavelet_buffer_key():
        from compressai.models._helpers.auxt import is_auxt_wavelet_buffer_key

        assert is_auxt_wavelet_buffer_key("AuxT_enc.0.dwt.transform.h0_col")
        assert is_auxt_wavelet_buffer_key("AuxT_dec.0.idwt.inverse.g0_col")
        assert not is_auxt_wavelet_buffer_key("AuxT_enc.0.olp.linear.weight")
        assert not is_auxt_wavelet_buffer_key("g_a.0.weight")

    @staticmethod
    def test_is_auxt_upstream_wavelet_buffer_key():
        from compressai.models._helpers.auxt import (
            is_auxt_upstream_wavelet_buffer_key,
        )

        for suffix in ("w_ll", "w_lh", "w_hl", "w_hh"):
            assert is_auxt_upstream_wavelet_buffer_key(f"AuxT_enc.0.dwt.{suffix}")
        assert is_auxt_upstream_wavelet_buffer_key("AuxT_dec.0.idwt.filters")
        assert not is_auxt_upstream_wavelet_buffer_key(
            "AuxT_enc.0.dwt.transform.h0_col"
        )
        assert not is_auxt_upstream_wavelet_buffer_key("AuxT_enc.0.olp.linear.weight")

    @staticmethod
    def test_normalize_upstream_auxt_key_renames_pascal_olp():
        from compressai.models._helpers.auxt import normalize_upstream_auxt_key

        assert (
            normalize_upstream_auxt_key("AuxT_enc.0.OLP.linear.weight")
            == "AuxT_enc.0.olp.linear.weight"
        )
        assert (
            normalize_upstream_auxt_key("AuxT_dec.3.OLP.linear.bias")
            == "AuxT_dec.3.olp.linear.bias"
        )
        # Returns None for non-AuxT keys so callers can use a single check.
        assert normalize_upstream_auxt_key("g_a.0.weight") is None
