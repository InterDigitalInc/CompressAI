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

import importlib.util

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.models.google import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    MeanScaleHyperprior,
    ScaleHyperprior,
    get_scale_table,
)
from compressai.models.utils import (
    _update_registered_buffer,
    find_named_module,
    update_registered_buffers,
)
from compressai.models.vbr import ScaleHyperpriorVbr
from compressai.models.video.google import ScaleSpaceFlow

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _load_convert_fn(script_name: str, fn_name: str):
    """Load a ``convert_upstream_*_state_dict`` function from an
    ``examples/convert_*_checkpoint.py`` script.

    The upstream-checkpoint conversion helpers live in the example CLI
    scripts (not in ``compressai.models.*``) so the model modules stay
    clean compressai-native definitions. ``examples/`` is not an importable
    package, so we load the module by file path.
    """
    path = _EXAMPLES_DIR / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, fn_name)


class DummyCompressionModel(CompressionModel):
    def __init__(self, entropy_bottleneck_channels):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)


class TestCompressionModel:
    def test_parameters(self):
        model = DummyCompressionModel(32)
        assert len(list(model.parameters())) == 15
        with pytest.raises(NotImplementedError):
            model(torch.rand(1))

    def test_init(self):
        class Model(DummyCompressionModel):
            def __init__(self):
                super().__init__(3)
                self.conv = nn.Conv2d(3, 3, 3)
                self.deconv = nn.ConvTranspose2d(3, 3, 3)
                self.original_conv = self.conv.weight
                self.original_deconv = self.deconv.weight

        model = Model()
        nn.init.kaiming_normal_(model.original_conv)
        nn.init.kaiming_normal_(model.original_deconv)

        assert torch.allclose(model.original_conv, model.conv.weight)
        assert torch.allclose(model.original_deconv, model.deconv.weight)


class TestModels:
    def test_factorized_prior(self):
        model = FactorizedPrior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

    def test_scale_hyperprior(self, tmpdir):
        model = ScaleHyperprior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperprior(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = ScaleHyperprior.from_state_dict(torch.load(filepath))
            assert model.N == loaded.N and model.M == loaded.M

    def test_scale_hyperprior_vbr(self, tmpdir):
        model = ScaleHyperpriorVbr(128, 192, vr_entbttlnck=True)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperpriorVbr(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = ScaleHyperpriorVbr.from_state_dict(torch.load(filepath))
            assert model.N == loaded.N and model.M == loaded.M

    def test_mean_scale_hyperprior(self):
        model = MeanScaleHyperprior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

    def test_jarhp(self, tmpdir):
        model = JointAutoregressiveHierarchicalPriors(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = JointAutoregressiveHierarchicalPriors(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = JointAutoregressiveHierarchicalPriors.from_state_dict(
                torch.load(filepath)
            )
            assert model.N == loaded.N and model.M == loaded.M

    def test_scale_space_flow(self):
        model = ScaleSpaceFlow()
        x = [torch.rand(1, 3, 128, 128), torch.rand(1, 3, 128, 128)]
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "keyframe" in out["likelihoods"][0]
        assert "y" in out["likelihoods"][0]["keyframe"]
        assert "z" in out["likelihoods"][0]["keyframe"]

        assert "motion" in out["likelihoods"][1]
        assert "y" in out["likelihoods"][1]["motion"]
        assert "z" in out["likelihoods"][1]["motion"]

        assert "residual" in out["likelihoods"][1]
        assert "y" in out["likelihoods"][1]["residual"]
        assert "z" in out["likelihoods"][1]["residual"]

        assert out["x_hat"][0].shape == x[0].shape
        assert out["x_hat"][1].shape == x[1].shape

        y_likelihoods_shape = out["likelihoods"][0]["keyframe"]["y"].shape
        assert y_likelihoods_shape[0] == x[0].shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x[0].shape[2] / 2**4
        assert y_likelihoods_shape[3] == x[0].shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"][0]["keyframe"]["z"].shape
        assert z_likelihoods_shape[0] == x[0].shape[0]
        assert z_likelihoods_shape[1] == 192
        assert z_likelihoods_shape[2] == x[0].shape[2] / 2**7  # (128x128 input)
        assert z_likelihoods_shape[3] == x[0].shape[3] / 2**7

        y_likelihoods_shape = out["likelihoods"][1]["motion"]["y"].shape
        assert y_likelihoods_shape[0] == x[1].shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x[1].shape[2] / 2**4
        assert y_likelihoods_shape[3] == x[1].shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"][1]["motion"]["z"].shape
        assert z_likelihoods_shape[0] == x[1].shape[0]
        assert z_likelihoods_shape[1] == 192
        assert z_likelihoods_shape[2] == x[1].shape[2] / 2**7  # (128x128 input)
        assert z_likelihoods_shape[3] == x[1].shape[3] / 2**7

        y_likelihoods_shape = out["likelihoods"][1]["residual"]["y"].shape
        assert y_likelihoods_shape[0] == x[1].shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x[1].shape[2] / 2**4
        assert y_likelihoods_shape[3] == x[1].shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"][1]["residual"]["z"].shape
        assert z_likelihoods_shape[0] == x[1].shape[0]
        assert z_likelihoods_shape[1] == 192
        assert z_likelihoods_shape[2] == x[1].shape[2] / 2**7  # (128x128 input)
        assert z_likelihoods_shape[3] == x[1].shape[3] / 2**7


class TestStf:
    def test_wacnn_forward_and_state_dict_round_trip(self):
        from compressai.models.stf import WACNN

        model = WACNN(N=64, M=128, num_slices=4, max_support_slices=2).eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        # Containerised state-dict layout self-check.
        sd_keys = set(model.state_dict().keys())
        assert "latent_codec.h_a.0.weight" in sd_keys
        assert "latent_codec.h_s.h_mean_s.0.weight" in sd_keys
        assert "latent_codec.h_s.h_scale_s.0.weight" in sd_keys
        assert "latent_codec.z.entropy_bottleneck.quantiles" in sd_keys
        # Side-parameter channel-context covers y0..y(K-1).
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in sd_keys
        assert "latent_codec.y.channel_context.y1.mean_cc.0.weight" in sd_keys
        assert "latent_codec.y.channel_context.y0.scale_cc.0.weight" in sd_keys
        # Per-slice leaves (LRP + per-slice GaussianConditional copy).
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in sd_keys
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table" in sd_keys
        )
        # Old monolithic paths should be gone.
        assert not any(
            k.startswith("latent_codec.cc_mean_transforms.") for k in sd_keys
        )
        assert "h_a.0.weight" not in sd_keys  # moved under latent_codec.

        loaded = WACNN.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])

    def test_symmetrical_transformer_forward_and_state_dict_round_trip(self):
        from compressai.models.stf import SymmetricalTransFormer

        model = SymmetricalTransFormer(
            embed_dim=24,
            depths=(1, 1, 1, 1),
            num_heads=(2, 2, 2, 2),
            num_slices=4,
            max_support_slices=2,
        ).eval()
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        sd_keys = set(model.state_dict().keys())
        assert "latent_codec.h_a.0.weight" in sd_keys
        assert "latent_codec.h_s.h_mean_s.0.weight" in sd_keys
        assert "latent_codec.z.entropy_bottleneck.quantiles" in sd_keys
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in sd_keys
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in sd_keys

        loaded = SymmetricalTransFormer.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])

    def test_stf_upstream_state_dict_conversion(self):
        convert_upstream_stf_state_dict = _load_convert_fn(
            "convert_stf_checkpoint.py", "convert_upstream_stf_state_dict"
        )

        upstream = {
            "module.g_a.0.weight": torch.zeros(2),
            "module.cc_mean_transforms.0.0.weight": torch.zeros(2),
            "module.cc_mean_transforms.1.0.weight": torch.zeros(2),
            "module.cc_scale_transforms.0.0.weight": torch.zeros(2),
            "module.lrp_transforms.0.0.weight": torch.zeros(2),
            "module.gaussian_conditional.scale_table": torch.zeros(2),
            "module.h_a.0.weight": torch.zeros(2),
            "module.h_mean_s.0.weight": torch.zeros(2),
            "module.h_scale_s.0.weight": torch.zeros(2),
            "module.entropy_bottleneck.quantiles": torch.zeros(2),
        }
        converted = convert_upstream_stf_state_dict(upstream)
        # g_a passes through unchanged.
        assert "g_a.0.weight" in converted
        # Hyperprior backbone moves under latent_codec.
        assert "latent_codec.h_a.0.weight" in converted
        assert "latent_codec.h_s.h_mean_s.0.weight" in converted
        assert "latent_codec.h_s.h_scale_s.0.weight" in converted
        assert "latent_codec.z.entropy_bottleneck.quantiles" in converted
        # cc_mean / cc_scale re-rooted per slice.
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in converted
        assert "latent_codec.y.channel_context.y1.mean_cc.0.weight" in converted
        assert "latent_codec.y.channel_context.y0.scale_cc.0.weight" in converted
        # gaussian_conditional replicated to every slice (driven by mean_cc count).
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y1.gaussian_conditional.scale_table"
            in converted
        )
        # LRP weights are now retained: emit_mean_support=True on the head
        # makes the leaf consume cat(latent_means, *prev_y_hat) as the LRP
        # input, matching upstream's M + slice_ch*(support+1) input width.
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in converted
        # Old root-level paths should be gone after conversion.
        assert "h_a.0.weight" not in converted
        assert "cc_mean_transforms.0.0.weight" not in converted
        assert "lrp_transforms.0.0.weight" not in converted


class TestTcm:
    def test_tcm_forward_and_state_dict_round_trip(self):
        from compressai.models.tcm import TCM

        model = TCM(
            N=32,
            M=64,
            hyper_channels=48,
            num_slices=4,
            max_support_slices=2,
        ).eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        # Containerised state-dict layout self-check.
        sd_keys = set(model.state_dict().keys())
        # Hyperprior backbone moved under latent_codec.* (TCM's h_a / h_*_s
        # use ResidualBlockWithStride / ResidualBlockUpsample, so the first
        # learnable weight is conv1 / conv).
        assert "latent_codec.h_a.0.conv1.weight" in sd_keys
        assert "latent_codec.h_s.h_mean_s.0.conv.weight" in sd_keys
        assert "latent_codec.h_s.h_scale_s.0.conv.weight" in sd_keys
        assert "latent_codec.z.entropy_bottleneck.quantiles" in sd_keys
        # Side-parameter channel-context covers y0..y(K-1).
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in sd_keys
        assert "latent_codec.y.channel_context.y1.mean_cc.0.weight" in sd_keys
        assert "latent_codec.y.channel_context.y0.scale_cc.0.weight" in sd_keys
        # SWAtten support transforms (TCM-specific; absent on STF/WACNN).
        assert (
            "latent_codec.y.channel_context.y0.mean_support_transform.in_conv.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.channel_context.y0.scale_support_transform.in_conv.weight"
            in sd_keys
        )
        # Per-slice leaves (LRP + per-slice GaussianConditional copy).
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in sd_keys
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table" in sd_keys
        )
        # Old monolithic / pr-stf-wacnn paths should be gone.
        assert not any(
            k.startswith("latent_codec.cc_mean_transforms.") for k in sd_keys
        )
        assert not any(k.startswith("latent_codec.atten_mean.") for k in sd_keys)
        assert "h_a.0.conv1.weight" not in sd_keys  # moved under latent_codec.

        loaded = TCM.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])
        assert loaded.N == 32
        assert loaded.M == 64
        assert loaded.hyper_channels == 48
        assert loaded.num_slices == 4
        assert loaded.max_support_slices == 2

    def test_tcm_upstream_state_dict_conversion(self):
        convert_upstream_tcm_state_dict = _load_convert_fn(
            "convert_tcm_checkpoint.py", "convert_upstream_tcm_state_dict"
        )

        # Synthetic upstream LIC_TCM-style state_dict: DataParallel ``module.``
        # prefix, raw entropy heads at the root, the SWAtten ``nn.Sequential``
        # wrapper level (``atten_mean.{k}.0.``), and a ConvTransBlock attention
        # buffer in upstream layout (``.msa.relative_position_params``).
        upstream = {
            "module.g_a.0.conv1.weight": torch.zeros(2),
            "module.g_a.1.trans_block.msa.relative_position_params": torch.zeros(
                4, 15, 15
            ),
            "module.g_a.1.trans_block.msa.embedding_layer.weight": torch.zeros(2),
            "module.g_a.1.trans_block.ln1.weight": torch.zeros(2),
            "module.g_a.1.trans_block.mlp.0.weight": torch.zeros(2),
            "module.g_a.1.trans_block.mlp.2.weight": torch.zeros(2),
            "module.cc_mean_transforms.0.0.weight": torch.zeros(2),
            "module.cc_mean_transforms.1.0.weight": torch.zeros(2),
            "module.cc_scale_transforms.0.0.weight": torch.zeros(2),
            "module.atten_mean.0.0.in_conv.weight": torch.zeros(2),
            "module.atten_scale.0.0.in_conv.weight": torch.zeros(2),
            "module.lrp_transforms.0.0.weight": torch.zeros(2),
            "module.gaussian_conditional.scale_table": torch.zeros(2),
            "module.h_a.0.conv1.weight": torch.zeros(2),
            "module.h_mean_s.0.conv.weight": torch.zeros(2),
            "module.h_scale_s.0.conv.weight": torch.zeros(2),
            "module.entropy_bottleneck.quantiles": torch.zeros(2),
        }
        converted = convert_upstream_tcm_state_dict(upstream)

        # ``module.`` prefix gone; g_a / ConvTransBlock pass through with the
        # MSA / layer-name renames applied.
        assert "g_a.0.conv1.weight" in converted
        # ``relative_position_params`` -> ``relative_position_bias_table``
        # with shape permuted from (2*win-1, 2*win-1, num_heads) =
        # (15, 15, 4) into the flat (225, 4) layout.
        assert "g_a.1.trans_block.msa.attn.relative_position_bias_table" in converted
        assert converted[
            "g_a.1.trans_block.msa.attn.relative_position_bias_table"
        ].shape == (15 * 15, 4)
        # ``embedding_layer`` -> ``attn.qkv``.
        assert "g_a.1.trans_block.msa.attn.qkv.weight" in converted
        # ``ln1`` -> ``norm1``; ``mlp.0`` / ``mlp.2`` -> ``mlp.fc1`` / ``fc2``.
        assert "g_a.1.trans_block.norm1.weight" in converted
        assert "g_a.1.trans_block.mlp.fc1.weight" in converted
        assert "g_a.1.trans_block.mlp.fc2.weight" in converted

        # Hyperprior backbone moves under latent_codec.
        assert "latent_codec.h_a.0.conv1.weight" in converted
        assert "latent_codec.h_s.h_mean_s.0.conv.weight" in converted
        assert "latent_codec.h_s.h_scale_s.0.conv.weight" in converted
        assert "latent_codec.z.entropy_bottleneck.quantiles" in converted

        # cc_mean / cc_scale re-rooted per slice.
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in converted
        assert "latent_codec.y.channel_context.y1.mean_cc.0.weight" in converted
        assert "latent_codec.y.channel_context.y0.scale_cc.0.weight" in converted

        # SWAtten wrapper unwrapped: ``atten_mean.0.0.<...>`` ->
        # ``...mean_support_transform.<...>`` (no extra ``.0`` level).
        assert (
            "latent_codec.y.channel_context.y0.mean_support_transform.in_conv.weight"
            in converted
        )
        assert (
            "latent_codec.y.channel_context.y0.scale_support_transform.in_conv.weight"
            in converted
        )

        # gaussian_conditional replicated per slice (driven by mean_cc count).
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y1.gaussian_conditional.scale_table"
            in converted
        )

        # LRP weights retained byte-for-byte (emit_mean_support=True path).
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in converted

        # Old root-level paths should be gone after conversion.
        assert "h_a.0.conv1.weight" not in converted
        assert "cc_mean_transforms.0.0.weight" not in converted
        assert "atten_mean.0.0.in_conv.weight" not in converted
        assert "lrp_transforms.0.0.weight" not in converted
        assert "module.g_a.0.conv1.weight" not in converted

    def test_tcm_use_auxt_default_false(self):
        from compressai.models.tcm import TCM

        model = TCM(
            N=32,
            M=64,
            hyper_channels=48,
            num_slices=4,
            max_support_slices=2,
        )
        assert model.use_auxt is False
        assert not hasattr(model.g_a, "auxiliary_layers")
        assert not hasattr(model.g_s, "auxiliary_layers")
        # aux_loss remains a scalar without AuxT, carrying the base
        # EntropyBottleneck auxiliary loss.
        loss = model.aux_loss()
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_tcm_use_auxt_construction_and_forward(self):
        pytest.importorskip("pytorch_wavelets")
        from compressai.models.tcm import TCM

        model = TCM(
            N=32,
            M=64,
            hyper_channels=48,
            num_slices=4,
            max_support_slices=2,
            use_auxt=True,
        ).eval()
        assert model.use_auxt is True
        assert len(model.g_a.auxiliary_layers) == 4
        assert len(model.g_s.auxiliary_layers) == 4
        # Default config (2,2,2,2,2,2) -> 10-layer g_a / g_s with merge
        # positions (0, 3, 6, 9) and (2, 5, 8, 9) respectively.
        assert model.g_a.merge_positions == (0, 3, 6, 9)
        assert model.g_s.merge_positions == (2, 5, 8, 9)

        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape

        # Aggregated OLP regulariser is a finite scalar > 0 with random init.
        loss = model.aux_loss()
        assert loss.dim() == 0
        assert torch.isfinite(loss) and loss.item() > 0

    def test_tcm_use_auxt_state_dict_round_trip(self):
        pytest.importorskip("pytorch_wavelets")
        from compressai.models.tcm import TCM

        model = TCM(
            N=32,
            M=64,
            hyper_channels=48,
            num_slices=4,
            max_support_slices=2,
            use_auxt=True,
        ).eval()
        sd = model.state_dict()
        # AuxT submodule paths are present.
        auxt_keys = {
            k
            for k in sd
            if k.startswith(("g_a.auxiliary_layers.", "g_s.auxiliary_layers."))
        }
        assert any(".olp.linear.weight" in k for k in auxt_keys)
        assert any(".scaling_factors" in k for k in auxt_keys)
        assert "g_a.transform.0.conv1.weight" in sd

        loaded = TCM.from_state_dict(sd).eval()
        # use_auxt is auto-detected from the wrapper auxiliary keys.
        assert loaded.use_auxt is True
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            assert torch.allclose(model(x)["x_hat"], loaded(x)["x_hat"])

    def test_tcm_convert_strips_upstream_wavelet_buffers_and_renames_olp(self):
        convert_upstream_tcm_state_dict = _load_convert_fn(
            "convert_tcm_checkpoint.py", "convert_upstream_tcm_state_dict"
        )

        # Synthetic upstream LIC_TCM-with-AuxT key set: minimal entropy
        # backbone keys to drive num_slices inference, plus AuxT keys with
        # the upstream-style ``.OLP.`` submodule and ``w_*`` / ``filters``
        # custom DWT/IDWT kernel buffers that should get dropped.
        upstream = {
            "module.g_a.0.conv1.weight": torch.zeros(2),
            "module.cc_mean_transforms.0.0.weight": torch.zeros(2),
            "module.cc_scale_transforms.0.0.weight": torch.zeros(2),
            "module.lrp_transforms.0.0.weight": torch.zeros(2),
            "module.gaussian_conditional.scale_table": torch.zeros(2),
            "module.h_a.0.conv1.weight": torch.zeros(2),
            "module.h_mean_s.0.conv.weight": torch.zeros(2),
            "module.h_scale_s.0.conv.weight": torch.zeros(2),
            "module.entropy_bottleneck.quantiles": torch.zeros(2),
            # AuxT keys: .OLP. should be renamed to .olp., w_*/filters dropped.
            "module.AuxT_enc.0.OLP.linear.weight": torch.zeros(8, 12),
            "module.AuxT_enc.0.OLP.linear.bias": torch.zeros(8),
            "module.AuxT_enc.0.scaling_factors": torch.zeros(1, 1, 12),
            "module.AuxT_enc.0.dwt.w_ll": torch.zeros(2, 2),
            "module.AuxT_enc.0.dwt.w_lh": torch.zeros(2, 2),
            "module.AuxT_enc.0.dwt.w_hl": torch.zeros(2, 2),
            "module.AuxT_enc.0.dwt.w_hh": torch.zeros(2, 2),
            "module.AuxT_dec.0.OLP.linear.weight": torch.zeros(12, 8),
            "module.AuxT_dec.0.idwt.filters": torch.zeros(4, 4),
        }
        converted = convert_upstream_tcm_state_dict(upstream)

        # ``.OLP.`` -> ``.olp.`` rename, ``module.`` prefix gone.
        assert "g_a.auxiliary_layers.0.olp.linear.weight" in converted
        assert "g_a.auxiliary_layers.0.olp.linear.bias" in converted
        assert "g_s.auxiliary_layers.0.olp.linear.weight" in converted
        # scaling_factors carries through.
        assert "g_a.auxiliary_layers.0.scaling_factors" in converted
        # With AuxT present, the main transform becomes the wrapped transform.
        assert "g_a.transform.0.conv1.weight" in converted
        assert "g_a.0.conv1.weight" not in converted
        # Upstream-LIC_TCM-specific DWT/IDWT kernel buffers dropped.
        for suffix in ("w_ll", "w_lh", "w_hl", "w_hh"):
            assert not any(
                k.endswith(suffix) for k in converted
            ), f"upstream DWT buffer {suffix} should have been dropped"
        assert not any(k.endswith(".idwt.filters") for k in converted)
        # Upstream-style PascalCase OLP keys should be gone.
        assert "g_a.auxiliary_layers.0.OLP.linear.weight" not in converted


class TestDcae:
    def _tiny_kwargs(self):
        return dict(
            N=64,
            M=80,
            hyper_channels=64,
            num_slices=4,
            max_support_slices=2,
            feature_dims=(48, 64, 80),
            block_num=(1, 1, 2),
            head_dim=(8, 8, 8, 8, 8, 8),
            dict_num=8,
            dict_head_num=4,
            dictionary_dim=32,
            window_size=4,
            hyper_window_size=2,
            hyper_head_dim=8,
        )

    def test_dcae_forward_and_state_dict_round_trip(self):
        from compressai.models.dcae import DCAE

        model = DCAE(**self._tiny_kwargs()).eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        sd_keys = set(model.state_dict().keys())
        # Shared dictionary lives at the model level (single state-dict path).
        assert "shared_dictionary.dt" in sd_keys
        assert sum(1 for k in sd_keys if k.endswith(".dt")) == 1
        # Hyperprior backbone moved under latent_codec.* (DCAE h_a wraps a
        # ResidualBottleneckBlockWithStride: outermost weight is .conv).
        assert "latent_codec.h_a.0.conv.weight" in sd_keys
        assert "latent_codec.h_s.h_mean_s.0.weight" in sd_keys
        assert "latent_codec.h_s.h_scale_s.0.weight" in sd_keys
        assert "latent_codec.z.entropy_bottleneck.quantiles" in sd_keys
        # Side-parameter channel context covers y0..y(K-1).
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in sd_keys
        assert "latent_codec.y.channel_context.y1.mean_cc.0.weight" in sd_keys
        # DCAE-specific dictionary cross-attention head.
        assert (
            "latent_codec.y.channel_context.y0.cross_attention.x_trans.weight"
            in sd_keys
        )
        # Per-slice leaves (LRP + per-slice GaussianConditional copy).
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in sd_keys
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table" in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y3.gaussian_conditional.scale_table" in sd_keys
        )
        # Old monolithic paths should be gone.
        assert "dt" not in sd_keys
        assert not any(k.startswith("dt_cross_attention.") for k in sd_keys)
        assert not any(k.startswith("cc_mean_transforms.") for k in sd_keys)
        assert not any(k.startswith("h_z_s1.") for k in sd_keys)
        assert not any(k.startswith("h_z_s2.") for k in sd_keys)

        loaded = DCAE.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])
        assert loaded.N == 64
        assert loaded.M == 80
        assert loaded.hyper_channels == 64
        assert loaded.num_slices == 4
        assert loaded.max_support_slices == 2
        assert loaded.dict_num == 8
        assert loaded.dict_head_num == 4
        assert loaded.dictionary_dim == 32

    def test_dcae_upstream_state_dict_conversion(self):
        convert_upstream_dcae_state_dict = _load_convert_fn(
            "convert_dcae_checkpoint.py", "convert_upstream_dcae_state_dict"
        )

        # Synthetic upstream DCAE-style state_dict: top-level dt + per-slice
        # ModuleLists for cc_mean / cc_scale / lrp / dt_cross_attention,
        # single shared gaussian_conditional, model-owned hyperprior with
        # h_z_s1 (scales) / h_z_s2 (means).
        m = 80
        # num_slices=4 -> slice_ch = m // 4 = 20 (used inline in shape calculations below)
        # cc_mean.0 first conv input width = M*3 + slice_ch * 0 = 240
        # cc_mean.1 first conv input width = M*3 + slice_ch * 1 = 260
        # lrp.0 first conv input width = M*3 + 0 + slice_ch = 260
        # cross_attention.0 input width = M*2 + 0 = 160
        # cross_attention.1 input width = M*2 + slice_ch = 180
        upstream = {
            # Top-level dictionary tensor.
            "dt": torch.zeros(8, 32),
            # Per-slice dt_cross_attention: x_trans is the only Linear that
            # consumes the original [scales, means, ...] input order.
            "dt_cross_attention.0.x_trans.weight": torch.arange(32 * 160)
            .float()
            .reshape(32, 160),
            "dt_cross_attention.0.scale": torch.zeros(4, 1, 1),
            "dt_cross_attention.1.x_trans.weight": torch.arange(32 * 180)
            .float()
            .reshape(32, 180),
            "dt_cross_attention.1.scale": torch.zeros(4, 1, 1),
            # Per-slice cc_mean / cc_scale: first conv has the means/scales swap.
            "cc_mean_transforms.0.0.weight": torch.arange(64 * 240)
            .float()
            .reshape(64, 240, 1, 1),
            "cc_mean_transforms.1.0.weight": torch.zeros(64, 260, 1, 1),
            "cc_scale_transforms.0.0.weight": torch.arange(64 * 240)
            .float()
            .reshape(64, 240, 1, 1),
            "cc_scale_transforms.1.0.weight": torch.zeros(64, 260, 1, 1),
            # Per-slice LRP: first conv also has the means/scales swap.
            "lrp_transforms.0.0.weight": torch.arange(64 * 260)
            .float()
            .reshape(64, 260, 1, 1),
            "lrp_transforms.1.0.weight": torch.zeros(64, 280, 1, 1),
            # Single shared gaussian_conditional (gets fanned out per slice).
            "gaussian_conditional.scale_table": torch.zeros(2),
            # Model-owned hyperprior backbone.
            "h_a.0.conv.weight": torch.zeros(64, 80, 5, 5),
            "h_z_s1.0.weight": torch.zeros(64, 64, 3, 3),  # scales (originally h_z_s1)
            "h_z_s2.0.weight": torch.zeros(64, 64, 3, 3),  # means (originally h_z_s2)
            "entropy_bottleneck.quantiles": torch.zeros(64, 1, 3),
            # g_a / g_s carry through unchanged.
            "g_a.0.conv.weight": torch.zeros(48, 3, 5, 5),
        }
        converted = convert_upstream_dcae_state_dict(upstream)

        # Top-level dt -> shared_dictionary.dt.
        assert "shared_dictionary.dt" in converted
        assert "dt" not in converted

        # Per-slice cross_attention re-rooted; x_trans.weight has its first 2*M
        # input channels (dim=1) swapped (means/scales reorder).
        assert (
            "latent_codec.y.channel_context.y0.cross_attention.x_trans.weight"
            in converted
        )
        original = upstream["dt_cross_attention.0.x_trans.weight"]
        swapped = converted[
            "latent_codec.y.channel_context.y0.cross_attention.x_trans.weight"
        ]
        # Swap should leave the trailing channels (>=2*M) unchanged but flip
        # the leading [0:M] and [M:2M] blocks.
        assert torch.equal(swapped[:, :m], original[:, m : 2 * m])
        assert torch.equal(swapped[:, m : 2 * m], original[:, :m])
        # cross_attention scale (head_num,1,1) carries through unchanged.
        assert "latent_codec.y.channel_context.y0.cross_attention.scale" in converted

        # Per-slice cc_mean / cc_scale re-rooted with means/scales swap on first conv.
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in converted
        original_cc = upstream["cc_mean_transforms.0.0.weight"]
        swapped_cc = converted["latent_codec.y.channel_context.y0.mean_cc.0.weight"]
        assert torch.equal(swapped_cc[:, :m], original_cc[:, m : 2 * m])
        assert torch.equal(swapped_cc[:, m : 2 * m], original_cc[:, :m])
        assert "latent_codec.y.channel_context.y1.mean_cc.0.weight" in converted
        assert "latent_codec.y.channel_context.y0.scale_cc.0.weight" in converted

        # Per-slice LRP re-rooted with means/scales swap on first conv.
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in converted
        original_lrp = upstream["lrp_transforms.0.0.weight"]
        swapped_lrp = converted["latent_codec.y.latent_codec.y0.lrp_transform.0.weight"]
        assert torch.equal(swapped_lrp[:, :m], original_lrp[:, m : 2 * m])
        assert torch.equal(swapped_lrp[:, m : 2 * m], original_lrp[:, :m])

        # gaussian_conditional fanned out to all K slices (driven by num_slices = 2 here:
        # only cc_mean has indices 0 and 1).
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y1.gaussian_conditional.scale_table"
            in converted
        )

        # Hyperprior backbone moved under latent_codec.*; h_z_s2 -> h_mean_s,
        # h_z_s1 -> h_scale_s (originally swapped on the upstream side).
        assert "latent_codec.h_a.0.conv.weight" in converted
        assert "latent_codec.h_s.h_mean_s.0.weight" in converted  # was h_z_s2
        assert "latent_codec.h_s.h_scale_s.0.weight" in converted  # was h_z_s1
        assert "latent_codec.z.entropy_bottleneck.quantiles" in converted

        # g_a / g_s carry through unchanged.
        assert "g_a.0.conv.weight" in converted

        # Old root-level paths should be gone after conversion.
        assert "h_a.0.conv.weight" not in converted
        assert "h_z_s1.0.weight" not in converted
        assert "h_z_s2.0.weight" not in converted
        assert "entropy_bottleneck.quantiles" not in converted
        assert "cc_mean_transforms.0.0.weight" not in converted
        assert "lrp_transforms.0.0.weight" not in converted
        assert "dt_cross_attention.0.x_trans.weight" not in converted
        assert "gaussian_conditional.scale_table" not in converted


class TestSaaf:
    def _tiny_kwargs(self):
        # Same shape as TestDcae for direct cross-comparison.
        return dict(
            N=64,
            M=80,
            hyper_channels=64,
            num_slices=4,
            max_support_slices=2,
            feature_dims=(48, 64, 80),
            block_num=(1, 1, 2),
            head_dim=(8, 8, 8, 8, 8, 8),
            dict_num=8,
            dict_head_num=4,
            dictionary_dim=32,
            window_size=4,
            hyper_window_size=2,
            hyper_head_dim=8,
        )

    def test_saaf_forward_and_state_dict_round_trip(self):
        from compressai.models.saaf import SAAF

        model = SAAF(**self._tiny_kwargs()).eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        # diffusion_loss is always present in the output dict; zero in eval mode.
        assert "diffusion_loss" in out
        assert out["diffusion_loss"].dim() == 0
        assert out["diffusion_loss"].item() == 0.0

        sd_keys = set(model.state_dict().keys())
        # Shared dictionary lives at the model level (single state-dict path).
        assert "shared_dictionary.dt" in sd_keys
        assert sum(1 for k in sd_keys if k.endswith(".dt")) == 1
        # Hyperprior backbone moved under latent_codec.* (SAAF h_a wraps a
        # ResidualBottleneckBlockWithStride: outermost weight is .conv).
        assert "latent_codec.h_a.0.conv.weight" in sd_keys
        assert "latent_codec.h_s.h_mean_s.0.weight" in sd_keys
        assert "latent_codec.h_s.h_scale_s.0.weight" in sd_keys
        assert "latent_codec.z.entropy_bottleneck.quantiles" in sd_keys
        # Side-parameter channel context covers y0..y(K-1).
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in sd_keys
        assert "latent_codec.y.channel_context.y1.mean_cc.0.weight" in sd_keys
        # Dictionary cross-attention head (shared with DCAE).
        assert (
            "latent_codec.y.channel_context.y0.cross_attention.x_trans.weight"
            in sd_keys
        )
        # Per-slice leaves (LRP + per-slice GaussianConditional copy).
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in sd_keys
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table" in sd_keys
        )
        # SAAF-specific: aux_enc / aux_dec each carry an OLP per stage,
        # and diffusion_prior holds the noise predictor.
        assert "aux_enc.0.olp.linear.weight" in sd_keys
        assert "aux_dec.3.olp.linear.weight" in sd_keys
        assert "diffusion_prior.noise_predictor.0.weight" in sd_keys
        # Old monolithic paths should be gone.
        assert "dt" not in sd_keys
        assert not any(k.startswith("dt_cross_attention.") for k in sd_keys)
        assert not any(k.startswith("cc_mean_transforms.") for k in sd_keys)
        assert not any(k.startswith("h_z_s1.") for k in sd_keys)
        assert not any(k.startswith("h_z_s2.") for k in sd_keys)

        loaded = SAAF.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])
        assert loaded.N == 64
        assert loaded.M == 80
        assert loaded.dict_num == 8

    def test_saaf_aux_loss_is_nonzero_scalar(self):
        from compressai.models.saaf import SAAF

        # SAAF integrates AuxT unconditionally (every _AdaptiveFrequencyBlock /
        # _InverseAdaptiveFrequencyBlock carries an OLP), so aux_loss is
        # always a non-trivial scalar — unlike TCM where use_auxt=False
        # gives zero.
        model = SAAF(**self._tiny_kwargs()).eval()
        loss = model.aux_loss()
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_saaf_diffusion_loss_active_in_training_mode(self):
        from compressai.models.saaf import SAAF

        model = SAAF(**self._tiny_kwargs()).train()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["diffusion_loss"].dim() == 0
        # Random init + random noise -> finite, non-zero scalar.
        assert torch.isfinite(out["diffusion_loss"])
        assert out["diffusion_loss"].item() > 0

    def test_saaf_upstream_state_dict_conversion(self):
        convert_upstream_saaf_state_dict = _load_convert_fn(
            "convert_saaf_checkpoint.py", "convert_upstream_saaf_state_dict"
        )

        # Synthetic upstream SAAF-style state_dict — same entropy backbone
        # as DCAE plus SAAF-specific aux_enc / aux_dec / diffusion_prior
        # keys that must pass through unchanged.
        m = 80
        # num_slices=4 -> slice_ch = m // 4 = 20 (used inline below)
        upstream = {
            "dt": torch.zeros(8, 32),
            "dt_cross_attention.0.x_trans.weight": torch.arange(32 * 160)
            .float()
            .reshape(32, 160),
            "dt_cross_attention.0.scale": torch.zeros(4, 1, 1),
            "cc_mean_transforms.0.0.weight": torch.arange(64 * 240)
            .float()
            .reshape(64, 240, 1, 1),
            "cc_mean_transforms.1.0.weight": torch.zeros(64, 260, 1, 1),
            "cc_scale_transforms.0.0.weight": torch.arange(64 * 240)
            .float()
            .reshape(64, 240, 1, 1),
            "lrp_transforms.0.0.weight": torch.arange(64 * 260)
            .float()
            .reshape(64, 260, 1, 1),
            "gaussian_conditional.scale_table": torch.zeros(2),
            "h_a.0.conv.weight": torch.zeros(64, 80, 5, 5),
            "h_z_s1.0.weight": torch.zeros(64, 64, 3, 3),  # scales
            "h_z_s2.0.weight": torch.zeros(64, 64, 3, 3),  # means
            "entropy_bottleneck.quantiles": torch.zeros(64, 1, 3),
            "g_a.0.conv.weight": torch.zeros(48, 3, 5, 5),
            # SAAF-specific keys that should pass through unchanged.
            "aux_enc.0.olp.linear.weight": torch.zeros(48, 3),
            "aux_enc.0.freq_weights": torch.zeros(4),
            "aux_dec.3.olp.linear.weight": torch.zeros(3, 48),
            "diffusion_prior.noise_predictor.0.weight": torch.zeros(80, 80, 3, 3),
        }
        converted = convert_upstream_saaf_state_dict(upstream)

        # Top-level dt -> shared_dictionary.dt.
        assert "shared_dictionary.dt" in converted
        assert "dt" not in converted

        # Means/scales swap on cross_attention.x_trans (first 2*M cols).
        original = upstream["dt_cross_attention.0.x_trans.weight"]
        swapped = converted[
            "latent_codec.y.channel_context.y0.cross_attention.x_trans.weight"
        ]
        assert torch.equal(swapped[:, :m], original[:, m : 2 * m])
        assert torch.equal(swapped[:, m : 2 * m], original[:, :m])

        # Same swap on cc_mean and lrp_transform first conv weights.
        for src_key, dst_key in (
            (
                "cc_mean_transforms.0.0.weight",
                "latent_codec.y.channel_context.y0.mean_cc.0.weight",
            ),
            (
                "lrp_transforms.0.0.weight",
                "latent_codec.y.latent_codec.y0.lrp_transform.0.weight",
            ),
        ):
            original_w = upstream[src_key]
            swapped_w = converted[dst_key]
            assert torch.equal(swapped_w[:, :m], original_w[:, m : 2 * m])
            assert torch.equal(swapped_w[:, m : 2 * m], original_w[:, :m])

        # gaussian_conditional fanned out per slice.
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y1.gaussian_conditional.scale_table"
            in converted
        )

        # h_z_s2 -> h_mean_s, h_z_s1 -> h_scale_s renames (DCAE convention).
        assert "latent_codec.h_a.0.conv.weight" in converted
        assert "latent_codec.h_s.h_mean_s.0.weight" in converted
        assert "latent_codec.h_s.h_scale_s.0.weight" in converted
        assert "latent_codec.z.entropy_bottleneck.quantiles" in converted

        # SAAF-specific aux_enc / aux_dec / diffusion_prior pass through
        # without any rename.
        assert "aux_enc.0.olp.linear.weight" in converted
        assert "aux_enc.0.freq_weights" in converted
        assert "aux_dec.3.olp.linear.weight" in converted
        assert "diffusion_prior.noise_predictor.0.weight" in converted

        # g_a / g_s pass through unchanged.
        assert "g_a.0.conv.weight" in converted

        # Old root-level entropy paths gone.
        assert "h_a.0.conv.weight" not in converted
        assert "h_z_s1.0.weight" not in converted
        assert "h_z_s2.0.weight" not in converted
        assert "entropy_bottleneck.quantiles" not in converted
        assert "cc_mean_transforms.0.0.weight" not in converted
        assert "lrp_transforms.0.0.weight" not in converted
        assert "dt_cross_attention.0.x_trans.weight" not in converted
        assert "gaussian_conditional.scale_table" not in converted


class TestMlic:
    def test_mlic_forward_and_state_dict_round_trip(self):
        pytest.importorskip("timm")
        from compressai.models.mlic import MLIC

        model = MLIC(N=8, M=12, slice_num=3, local_kernel=3).eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        sd_keys = set(model.state_dict().keys())
        assert "latent_codec.h_a.reduction.0.weight" in sd_keys
        assert "latent_codec.h_s.increase.0.weight" in sd_keys
        assert "latent_codec.z.entropy_bottleneck.quantiles" in sd_keys
        assert (
            "latent_codec.y.channel_context.y1.channel_part.fushion.0.weight" in sd_keys
        )
        assert not any(
            k.startswith("latent_codec.y.channel_context.y1.global_inter_part.")
            for k in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y0.spatial_context_nonanchor.context.0.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y1.intra_channel_context_nonanchor.keys.0.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y0.y.gaussian_conditional.scale_table"
            in sd_keys
        )
        assert "h_a.reduction.0.weight" not in sd_keys
        assert not hasattr(model, "entropy_bottleneck")
        assert not hasattr(model, "gaussian_conditional")

        loaded = MLIC.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])
        assert torch.allclose(out["likelihoods"]["y"], out_loaded["likelihoods"]["y"])
        assert torch.allclose(out["likelihoods"]["z"], out_loaded["likelihoods"]["z"])
        assert loaded.N == 8
        assert loaded.M == 12
        assert loaded.slice_num == 3
        assert loaded.local_kernel == 3
        assert loaded.local_layers == 3

    def test_mlicplus_forward_and_state_dict_round_trip(self):
        pytest.importorskip("timm")
        from compressai.models.mlic import MLICPlus

        model = MLICPlus(N=8, M=16, slice_num=4, context_window=3).eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        sd_keys = set(model.state_dict().keys())
        assert (
            "latent_codec.y.channel_context.y1.global_inter_part.keys.0.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y0.spatial_context_nonanchor.relative_position_table"
            in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y1.intra_channel_context_nonanchor.keys.0.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y0.lrp_anchor.lrp_transform.0.weight"
            in sd_keys
        )
        assert "latent_codec.entropy_bottleneck.quantiles" not in sd_keys

        loaded = MLICPlus.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])
        assert torch.allclose(out["likelihoods"]["y"], out_loaded["likelihoods"]["y"])
        assert torch.allclose(out["likelihoods"]["z"], out_loaded["likelihoods"]["z"])
        assert loaded.N == 8
        assert loaded.M == 16
        assert loaded.slice_num == 4
        assert loaded.context_window == 3

    def test_mlicpp_upstream_state_dict_conversion(self):
        convert_upstream_mlicpp_state_dict = _load_convert_fn(
            "convert_mlic_checkpoint.py", "convert_upstream_mlicpp_state_dict"
        )

        upstream = {
            "h_a.reduction.0.weight": torch.zeros(2),
            "h_s.increase.0.weight": torch.zeros(2),
            "entropy_bottleneck.quantiles": torch.zeros(2),
            "gaussian_conditional.scale_table": torch.zeros(2),
            "local_context.0.relative_position_table": torch.zeros(2),
            "channel_context.0.fushion.0.weight": torch.zeros(2),
            "global_inter_context.1.keys.0.weight": torch.zeros(2),
            "global_intra_context.1.keys.0.weight": torch.zeros(2),
            "entropy_parameters_anchor.0.fusion.0.weight": torch.zeros(2),
            "entropy_parameters_nonanchor.1.fusion.0.weight": torch.zeros(2),
            "lrp_anchor.0.lrp_transform.0.weight": torch.zeros(2),
            "lrp_nonanchor.1.lrp_transform.0.weight": torch.zeros(2),
        }
        converted = convert_upstream_mlicpp_state_dict(upstream)

        assert "latent_codec.h_a.reduction.0.weight" in converted
        assert "latent_codec.h_s.increase.0.weight" in converted
        assert "latent_codec.z.entropy_bottleneck.quantiles" in converted
        assert (
            "latent_codec.y.latent_codec.y0.y.gaussian_conditional.scale_table"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y1.y.gaussian_conditional.scale_table"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y0.spatial_context_nonanchor.relative_position_table"
            in converted
        )
        assert (
            "latent_codec.y.channel_context.y0.channel_part.fushion.0.weight"
            in converted
        )
        assert (
            "latent_codec.y.channel_context.y1.global_inter_part.keys.0.weight"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y1.intra_channel_context_nonanchor.keys.0.weight"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y0.entropy_parameters_anchor.fusion.0.weight"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y1.entropy_parameters_nonanchor.fusion.0.weight"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y0.lrp_anchor.lrp_transform.0.weight"
            in converted
        )
        assert (
            "latent_codec.y.latent_codec.y1.lrp_nonanchor.lrp_transform.0.weight"
            in converted
        )

        assert "h_a.reduction.0.weight" not in converted
        assert "entropy_bottleneck.quantiles" not in converted
        assert "gaussian_conditional.scale_table" not in converted
        assert "local_context.0.relative_position_table" not in converted
        assert "channel_context.0.fushion.0.weight" not in converted


class TestMlicv2:
    def test_forward_state_dict_round_trip_and_gsc_skip_rate(self):
        pytest.importorskip("timm")
        from compressai.models.mlic import MLICv2

        model = MLICv2(N=8, M=16, slice_num=4, context_window=3).eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        sd_keys = set(model.state_dict().keys())
        assert "g_a.analysis_transform.1.0.norm1.weight" in sd_keys
        assert "g_s.synthesis_transform.0.0.norm1.weight" in sd_keys
        assert (
            "latent_codec.y.latent_codec.y0.spatial_context_anchor.hgcp.queries.0.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y0.selective_predictor.predictor.0.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.channel_context.y1.global_inter_part.context.keys.0.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.channel_context.y1.global_inter_part.reweighting.queries.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y1.intra_channel_context_nonanchor.context.keys.0.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.latent_codec.y1.intra_channel_context_nonanchor.rope.theta_x"
            in sd_keys
        )
        assert "latent_codec.entropy_bottleneck.quantiles" not in sd_keys

        predictor = model.latent_codec.y.latent_codec["y0"].selective_predictor
        side_params = torch.randn(1, 32, 4, 4)
        scales = torch.linspace(0.1, 0.5, steps=1 * 4 * 4 * 4).reshape(1, 4, 4, 4)
        means = torch.zeros_like(scales)
        with torch.no_grad():
            selective = predictor(
                side_params=side_params,
                scales=scales,
                means=means,
                step="anchor",
            )["selective_map"]
        hard_ratio = (selective >= 0.5).float().mean().item()
        assert 0.1 < hard_ratio < 0.9

        loaded = MLICv2.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])
        assert torch.allclose(out["likelihoods"]["y"], out_loaded["likelihoods"]["y"])
        assert torch.allclose(out["likelihoods"]["z"], out_loaded["likelihoods"]["z"])
        assert loaded.N == 8
        assert loaded.M == 16
        assert loaded.slice_num == 4
        assert loaded.context_window == 3
        assert loaded.downsampling_factor == 64


class TestCca:
    def test_cca_forward_and_state_dict_round_trip(self):
        from compressai.models.cca import CCAModel

        # Tiny variant — variable-length slices, smaller dims keep the
        # NAFTransform stack cheap. Slice proportions reproduce the
        # upstream layout (8/28/56/92/136 over M=320) at scale.
        model = CCAModel(
            latent_channels=64,
            hyper_channels=48,
            slice_proportions=(2, 6, 12, 18, 26),
            encoder_dims=(48, 56, 64),
            encoder_layers=(1, 1, 1),
            em_hidden_channels=56,
            em_num_layers=1,
        ).eval()
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        # cca_training=False -> no aux likelihoods exposed.
        assert out["aux_likelihoods"] is None

        # Containerised state-dict layout self-check.
        sd_keys = set(model.state_dict().keys())
        # Hyperprior backbone moved under latent_codec.* (CCA's h_a /
        # h_*_s use plain Sequentials of conv / GELU; first weight is at
        # `.0.weight` rather than `.0.conv1.weight`).
        assert "latent_codec.h_a.0.weight" in sd_keys
        assert "latent_codec.h_s.h_mean_s.0.weight" in sd_keys
        assert "latent_codec.h_s.h_scale_s.0.weight" in sd_keys
        # STF/WACNN/TCM/CCA use STE on z; the
        # entropy_bottleneck still owns the parametric prior.
        assert "latent_codec.z.entropy_bottleneck.quantiles" in sd_keys
        # Side-parameter channel-context covers y0..y(K-1).
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in sd_keys
        assert "latent_codec.y.channel_context.y4.mean_cc.0.weight" in sd_keys
        assert "latent_codec.y.channel_context.y0.scale_cc.0.weight" in sd_keys
        # NAFTransform support transforms (CCA-specific; absent on STF/WACNN).
        assert (
            "latent_codec.y.channel_context.y0.mean_support_transform.input_projection.weight"
            in sd_keys
        )
        assert (
            "latent_codec.y.channel_context.y0.scale_support_transform.input_projection.weight"
            in sd_keys
        )
        # Per-slice leaves (LRP + per-slice GaussianConditional copy).
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in sd_keys
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table" in sd_keys
        )
        # Old monolithic / pre-refactor paths should be gone.
        assert "h_a.0.weight" not in sd_keys
        assert "z_entropy_bottleneck.quantiles" not in sd_keys
        assert not any(k.startswith("mean_cc_transforms.") for k in sd_keys)
        assert not any(k.startswith("aux_entropy_model.") for k in sd_keys)

        loaded = CCAModel.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])
        assert torch.allclose(out["likelihoods"]["y"], out_loaded["likelihoods"]["y"])
        assert torch.allclose(out["likelihoods"]["z"], out_loaded["likelihoods"]["z"])
        assert loaded.M == 64
        assert loaded.N == 48
        assert tuple(loaded.slice_sizes) == (2, 6, 12, 18, 26)
        assert loaded.em_hidden_channels == 56
        assert loaded.em_num_layers == 1
        assert loaded.cca_training is False

    def test_cca_training_branch_forward_and_round_trip(self):
        from compressai.models.cca import CCAModel

        model = CCAModel(
            latent_channels=64,
            hyper_channels=48,
            slice_proportions=(2, 6, 12, 18, 26),
            encoder_dims=(48, 56, 64),
            encoder_layers=(1, 1, 1),
            em_hidden_channels=56,
            em_num_layers=1,
            cca_training=True,
        ).eval()
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        # Aux branch populates y_aux (factorised) and y_cca (Gaussian).
        assert isinstance(out["aux_likelihoods"], dict)
        assert set(out["aux_likelihoods"].keys()) == {"y_aux", "y_cca"}
        assert out["aux_likelihoods"]["y_aux"].shape == out["likelihoods"]["y"].shape
        assert out["aux_likelihoods"]["y_cca"].shape == out["likelihoods"]["y"].shape

        # Aux state-dict paths (skip-most-recent inner ChannelGroupsLatentCodec).
        sd_keys = set(model.state_dict().keys())
        assert "aux_entropy_model.y_entropy_bottleneck.quantiles" in sd_keys
        assert (
            "aux_entropy_model.inner_codec.channel_context.y0.mean_cc.0.weight"
            in sd_keys
        )
        assert (
            "aux_entropy_model.inner_codec.channel_context.y0.mean_support_transform.input_projection.weight"
            in sd_keys
        )
        assert (
            "aux_entropy_model.inner_codec.latent_codec.y0.lrp_transform.0.weight"
            in sd_keys
        )

        loaded = CCAModel.from_state_dict(model.state_dict()).eval()
        with torch.no_grad():
            out_loaded = loaded(x)
        assert loaded.cca_training is True
        assert torch.allclose(
            out["aux_likelihoods"]["y_aux"], out_loaded["aux_likelihoods"]["y_aux"]
        )
        assert torch.allclose(
            out["aux_likelihoods"]["y_cca"], out_loaded["aux_likelihoods"]["y_cca"]
        )

    def test_cca_upstream_state_dict_conversion(self):
        convert_upstream_cca_state_dict = _load_convert_fn(
            "convert_cca_checkpoint.py", "convert_upstream_cca_state_dict"
        )
        _is_upstream_cca_state_dict = _load_convert_fn(
            "convert_cca_checkpoint.py", "_is_upstream_cca_state_dict"
        )

        # Synthetic upstream LICAutoencoder-style state_dict with one slice
        # per branch covering the full path: NAFBlock interior renames,
        # NAFTransform interior renames, named-part NAF -> support_transforms
        # alias, top-level hyperprior + aux module rerooting, per-slice
        # rerooting under channel_context / latent_codec, and the
        # gaussian_conditional replication.
        upstream = {
            # ResidualBottleneckBlock inside g_a (conv1 should NOT be renamed
            # since it's not inside a NAFBlock — checked via the NAFBlock
            # detector which requires the .beta/.gamma/.dwconv.0 triple).
            "g_a.blocks.0.0.conv1.weight": torch.zeros(2),
            # NAFBlock inside g_a (full triple present -> dwconv/sca/FFN/conv1
            # interior renames apply to this scope only).
            "g_a.blocks.0.3.beta": torch.zeros(2),
            "g_a.blocks.0.3.gamma": torch.zeros(2),
            "g_a.blocks.0.3.dwconv.0.weight": torch.zeros(2),
            "g_a.blocks.0.3.sca.1.weight": torch.zeros(2),
            "g_a.blocks.0.3.FFN.0.weight": torch.zeros(2),
            "g_a.blocks.0.3.conv1.weight": torch.zeros(2),
            # Per-slice main entropy heads (one slice for compactness).
            "mean_cc_transforms.0.0.weight": torch.zeros(2),
            "scale_cc_transforms.0.0.weight": torch.zeros(2),
            "lrp_transforms.0.0.weight": torch.zeros(2),
            # NAFTransform interior (in_conv/out_conv -> input_projection/...).
            # Triple required for the detector: .in_conv.weight,
            # .out_conv.weight, .blocks.0.beta.
            "mean_NAF_transforms.0.in_conv.weight": torch.zeros(2),
            "mean_NAF_transforms.0.out_conv.weight": torch.zeros(2),
            "mean_NAF_transforms.0.blocks.0.beta": torch.zeros(2),
            "scale_NAF_transforms.0.in_conv.weight": torch.zeros(2),
            "scale_NAF_transforms.0.out_conv.weight": torch.zeros(2),
            "scale_NAF_transforms.0.blocks.0.beta": torch.zeros(2),
            "gaussian_conditional.scale_table": torch.zeros(2),
            # Hyperprior backbone (root-level -> latent_codec.*).
            "h_a.0.weight": torch.zeros(2),
            "h_mean_s.0.weight": torch.zeros(2),
            "h_scale_s.0.weight": torch.zeros(2),
            "z_entropy_bottleneck.quantiles": torch.zeros(2),
            # Aux entropy module (aux_entropymodel -> aux_entropy_model, then
            # the same per-slice rerooting as the main path).
            "aux_entropymodel.mean_cc_transforms.0.0.weight": torch.zeros(2),
            "aux_entropymodel.scale_cc_transforms.0.0.weight": torch.zeros(2),
            "aux_entropymodel.lrp_transforms.0.0.weight": torch.zeros(2),
            "aux_entropymodel.mean_NAF_transforms.0.in_conv.weight": torch.zeros(2),
            "aux_entropymodel.mean_NAF_transforms.0.out_conv.weight": torch.zeros(2),
            "aux_entropymodel.mean_NAF_transforms.0.blocks.0.beta": torch.zeros(2),
            "aux_entropymodel.scale_NAF_transforms.0.in_conv.weight": torch.zeros(2),
            "aux_entropymodel.scale_NAF_transforms.0.out_conv.weight": torch.zeros(2),
            "aux_entropymodel.scale_NAF_transforms.0.blocks.0.beta": torch.zeros(2),
            "aux_entropymodel.gaussian_conditional.scale_table": torch.zeros(2),
            "aux_entropymodel.y_entropy_bottleneck.quantiles": torch.zeros(2),
        }
        assert _is_upstream_cca_state_dict(upstream)

        converted = convert_upstream_cca_state_dict(upstream)

        # ResidualBottleneckBlock conv1 NOT renamed (not inside NAFBlock).
        assert "g_a.blocks.0.0.conv1.weight" in converted
        # NAFBlock interior renames applied at the NAFBlock scope only.
        assert "g_a.blocks.0.3.beta" in converted
        assert "g_a.blocks.0.3.pointwise_depthwise.0.weight" in converted
        assert "g_a.blocks.0.3.channel_attention.1.weight" in converted
        assert "g_a.blocks.0.3.feed_forward.0.weight" in converted
        assert "g_a.blocks.0.3.project.weight" in converted

        # Hyperprior backbone moves under latent_codec.
        assert "latent_codec.h_a.0.weight" in converted
        assert "latent_codec.h_s.h_mean_s.0.weight" in converted
        assert "latent_codec.h_s.h_scale_s.0.weight" in converted
        assert "latent_codec.z.entropy_bottleneck.quantiles" in converted

        # Per-slice main rerooting.
        assert "latent_codec.y.channel_context.y0.mean_cc.0.weight" in converted
        assert "latent_codec.y.channel_context.y0.scale_cc.0.weight" in converted
        # NAFTransform: in_conv -> input_projection; mean_NAF_transforms ->
        # channel_context.y{k}.mean_support_transform.
        assert (
            "latent_codec.y.channel_context.y0.mean_support_transform.input_projection.weight"
            in converted
        )
        assert (
            "latent_codec.y.channel_context.y0.scale_support_transform.input_projection.weight"
            in converted
        )
        # gaussian_conditional replicated under each per-slice leaf.
        assert (
            "latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table"
            in converted
        )
        # LRP weights byte-for-byte under per-slice leaf.
        assert "latent_codec.y.latent_codec.y0.lrp_transform.0.weight" in converted

        # Aux entropy module rerooting (aux_entropymodel -> aux_entropy_model;
        # per-slice contents land under inner_codec.*).
        assert (
            "aux_entropy_model.inner_codec.channel_context.y0.mean_cc.0.weight"
            in converted
        )
        assert (
            "aux_entropy_model.inner_codec.channel_context.y0.mean_support_transform.input_projection.weight"
            in converted
        )
        assert (
            "aux_entropy_model.inner_codec.latent_codec.y0.lrp_transform.0.weight"
            in converted
        )
        assert (
            "aux_entropy_model.inner_codec.latent_codec.y0.gaussian_conditional.scale_table"
            in converted
        )
        assert "aux_entropy_model.y_entropy_bottleneck.quantiles" in converted

        # Old paths should be gone after conversion.
        assert "h_a.0.weight" not in converted
        assert "z_entropy_bottleneck.quantiles" not in converted
        assert "mean_cc_transforms.0.0.weight" not in converted
        assert "mean_NAF_transforms.0.in_conv.weight" not in converted
        assert "lrp_transforms.0.0.weight" not in converted
        assert "aux_entropymodel.mean_cc_transforms.0.0.weight" not in converted


def test_scale_table_default():
    table = get_scale_table()
    assert SCALES_MIN == 0.11
    assert SCALES_MAX == 256
    assert SCALES_LEVELS == 64
    assert table[0] == SCALES_MIN
    assert table[-1] == SCALES_MAX
    assert len(table.size()) == 1
    assert table.size(0) == SCALES_LEVELS


def test_scale_table_custom():
    table = get_scale_table(0.02, 1337, 32)
    assert pytest.approx(table[0].item()) == 0.02
    assert pytest.approx(table[-1].item()) == 1337
    assert len(table.size()) == 1
    assert table.size(0) == 32


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 1)


def test_find_named_module():
    assert find_named_module(Foo(), "conv3") is None
    foo = Foo()
    found = find_named_module(foo, "conv1")
    assert found == foo.conv1


def test_update_registered_buffers():
    foo = Foo()
    with pytest.raises(ValueError):
        update_registered_buffers(foo, "conv1", ["qweight"], {})


def test_update_registered_buffer():
    foo = Foo()

    # non-registered buffer
    state_dict = foo.state_dict()
    state_dict["conv1.wweight"] = torch.rand(3)
    with pytest.raises(RuntimeError):
        _update_registered_buffer(
            foo.conv1, "wweight", "conv1.wweight", state_dict, policy="resize"
        )
    with pytest.raises(RuntimeError):
        _update_registered_buffer(
            foo.conv1, "wweight", "conv1.wweight", state_dict, policy="resize_if_empty"
        )
