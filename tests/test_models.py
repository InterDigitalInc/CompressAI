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
