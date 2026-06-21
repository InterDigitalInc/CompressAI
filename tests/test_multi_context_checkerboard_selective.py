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

from typing import List, Optional

import torch
import torch.nn as nn

from compressai.latent_codecs import MultiContextCheckerboardLatentCodec
from compressai.layers import CheckerboardMaskedConv2d


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


class _StepSelectivePredictor(nn.Module):
    def __init__(self, *, anchor_value: float, non_anchor_value: float) -> None:
        super().__init__()
        self.anchor_value = float(anchor_value)
        self.non_anchor_value = float(non_anchor_value)

    def forward(
        self,
        *,
        side_params: torch.Tensor,
        scales: torch.Tensor,
        means: torch.Tensor,
        step: str,
    ) -> torch.Tensor:
        value = self.anchor_value if step == "anchor" else self.non_anchor_value
        return torch.full_like(scales, value)


def _scale_table() -> List[float]:
    return [0.11, 0.5, 1.0, 2.0, 4.0]


def _make(
    *,
    y_ch: int = 4,
    side_ch: int = 8,
    anchor_in: Optional[int] = None,
    nonanchor_in: Optional[int] = None,
    **kwargs,
) -> MultiContextCheckerboardLatentCodec:
    if anchor_in is None:
        anchor_in = side_ch
    if nonanchor_in is None:
        nonanchor_in = side_ch
    return MultiContextCheckerboardLatentCodec(
        entropy_parameters_anchor=nn.Conv2d(anchor_in, 2 * y_ch, 1),
        entropy_parameters_nonanchor=nn.Conv2d(nonanchor_in, 2 * y_ch, 1),
        scale_table=_scale_table(),
        **kwargs,
    )


def _non_anchor_mask_like(y: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[..., 0::2, 1::2] = True
    mask[..., 1::2, 0::2] = True
    return mask


def _encoded_size(strings: List[List[bytes]]) -> int:
    return sum(len(s) for step_strings in strings for s in step_strings)


def test_selective_predictor_none_is_identity() -> None:
    torch.manual_seed(23)
    kwargs = dict(
        nonanchor_in=8 + 4,
        spatial_context_nonanchor=CheckerboardMaskedConv2d(4, 4, 5, padding=2),
    )
    baseline = _make(**kwargs).eval()
    explicit_none = _make(selective_predictor=None, **kwargs).eval()
    explicit_none.load_state_dict(baseline.state_dict())
    baseline.y.gaussian_conditional.update()
    explicit_none.y.gaussian_conditional.update()

    y = torch.randn(1, 4, 8, 8)
    side_params = torch.randn(1, 8, 8, 8)

    with torch.no_grad():
        baseline_forward = baseline(y, side_params)
        explicit_forward = explicit_none(y, side_params)
    assert torch.allclose(
        baseline_forward["likelihoods"]["y"],
        explicit_forward["likelihoods"]["y"],
    )
    assert torch.allclose(baseline_forward["y_hat"], explicit_forward["y_hat"])

    baseline_compressed = baseline.compress(y, side_params)
    explicit_compressed = explicit_none.compress(y, side_params)
    assert baseline_compressed["strings"] == explicit_compressed["strings"]
    assert torch.allclose(baseline_compressed["y_hat"], explicit_compressed["y_hat"])

    baseline_decompressed = baseline.decompress(
        baseline_compressed["strings"], baseline_compressed["shape"], side_params
    )
    explicit_decompressed = explicit_none.decompress(
        explicit_compressed["strings"], explicit_compressed["shape"], side_params
    )
    assert torch.allclose(
        baseline_decompressed["y_hat"], explicit_decompressed["y_hat"]
    )


def test_selective_predictor_skip_semantics() -> None:
    selective = MultiContextCheckerboardLatentCodec(
        entropy_parameters_anchor=_ZeroEntropyParameters(8),
        entropy_parameters_nonanchor=_ZeroEntropyParameters(8),
        scale_table=_scale_table(),
        selective_predictor=_StepSelectivePredictor(
            anchor_value=1.0,
            non_anchor_value=0.0,
        ),
    ).eval()
    full = MultiContextCheckerboardLatentCodec(
        entropy_parameters_anchor=_ZeroEntropyParameters(8),
        entropy_parameters_nonanchor=_ZeroEntropyParameters(8),
        scale_table=_scale_table(),
    ).eval()
    selective.y.gaussian_conditional.update()
    full.y.gaussian_conditional.update()

    y = torch.linspace(-2.0, 2.0, steps=64).reshape(1, 4, 4, 4)
    side_params = torch.zeros(1, 8, 4, 4)
    non_anchor_mask = _non_anchor_mask_like(y)

    with torch.no_grad():
        out = selective(y, side_params)

    assert torch.all(out["likelihoods"]["y"][non_anchor_mask] == 1)
    assert torch.allclose(
        out["y_hat"][non_anchor_mask],
        torch.zeros_like(out["y_hat"][non_anchor_mask]),
    )

    selective_compressed = selective.compress(y, side_params)
    full_compressed = full.compress(y, side_params)
    assert selective_compressed["strings"][1] == [b""]
    assert _encoded_size(selective_compressed["strings"]) < _encoded_size(
        full_compressed["strings"]
    )

    decompressed = selective.decompress(
        selective_compressed["strings"],
        selective_compressed["shape"],
        side_params,
    )
    assert torch.allclose(selective_compressed["y_hat"], decompressed["y_hat"])
    assert torch.allclose(
        decompressed["y_hat"][non_anchor_mask],
        torch.zeros_like(decompressed["y_hat"][non_anchor_mask]),
    )
