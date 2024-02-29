# Copyright (c) 2021-2024, InterDigital Communications, Inc
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

from typing import Any, Dict, List, Mapping, Optional, Tuple

from torch import Tensor

from compressai.registry import register_module

from ..base import LatentCodec
from ..gaussian_conditional import GaussianConditionalLatentCodec
from .hyper import GainHyperLatentCodec

__all__ = [
    "GainHyperpriorLatentCodec",
]


@register_module("GainHyperpriorLatentCodec")
class GainHyperpriorLatentCodec(LatentCodec):
    """Hyperprior codec constructed from latent codec for ``y`` that
    compresses ``y`` using ``params`` from ``hyper`` branch.

    Gain-controlled hyperprior introduced in
    `"Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation"
    <https://arxiv.org/abs/2003.02012>`_, by Ze Cui, Jing Wang,
    Shangyin Gao, Bo Bai, Tiansheng Guo, and Yihui Feng, CVPR, 2021.

    .. code-block:: none

                z_gain  z_gain_inv
                   │        │
                   ▼        ▼
                  ┌┴────────┴┐
            ┌──►──┤ lc_hyper ├──►─┐
            │     └──────────┘    │
            │                     │
            │     y_gain          ▼ params   y_gain_inv
            │        │            │              │
            │        ▼            │              ▼
            │        │         ┌──┴───┐          │
        y ──┴────►───×───►─────┤ lc_y ├────►─────×─────►── y_hat
                               └──────┘

    By default, the following codec is constructed:

    .. code-block:: none

                        z_gain                      z_gain_inv
                           │                             │
                           ▼                             ▼
                 ┌───┐  z  │ z_g ┌───┐ z_hat      z_hat  │       ┌───┐
            ┌─►──┤h_a├──►──×──►──┤ Q ├───►───····───►────×────►──┤h_s├──┐
            │    └───┘           └───┘        EB                 └───┘  │
            │                                                           │
            │                              ┌──────────────◄─────────────┘
            │                              │            params
            │                           ┌──┴──┐
            │    y_gain                 │  EP │    y_gain_inv
            │       │                   └──┬──┘        │
            │       ▼                      │           ▼
            │       │       ┌───┐          ▼           │
        y ──┴───►───×───►───┤ Q ├────►────····───►─────×─────►── y_hat
                            └───┘          GC

    Common configurations of latent codecs include:
     - entropy bottleneck ``hyper`` (default) and gaussian conditional ``y`` (default)
     - entropy bottleneck ``hyper`` (default) and autoregressive ``y``
    """

    latent_codec: Mapping[str, LatentCodec]

    def __init__(
        self, latent_codec: Optional[Mapping[str, LatentCodec]] = None, **kwargs
    ):
        super().__init__()
        self._set_group_defaults(
            "latent_codec",
            latent_codec,
            defaults={
                "y": GaussianConditionalLatentCodec,
                "hyper": GainHyperLatentCodec,
            },
            save_direct=True,
        )

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(
        self,
        y: Tensor,
        y_gain: Tensor,
        z_gain: Tensor,
        y_gain_inv: Tensor,
        z_gain_inv: Tensor,
    ) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"](y, z_gain, z_gain_inv)
        y_out = self.latent_codec["y"](y * y_gain, hyper_out["params"])
        y_hat = y_out["y_hat"] * y_gain_inv
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
            "y_hat": y_hat,
        }

    def compress(
        self,
        y: Tensor,
        y_gain: Tensor,
        z_gain: Tensor,
        y_gain_inv: Tensor,
        z_gain_inv: Tensor,
    ) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"].compress(y, z_gain, z_gain_inv)
        y_out = self.latent_codec["y"].compress(y * y_gain, hyper_out["params"])
        y_hat = y_out["y_hat"] * y_gain_inv
        return {
            "strings": [*y_out["strings"], *hyper_out["strings"]],
            "shape": {"y": y_out["shape"], "hyper": hyper_out["shape"]},
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Dict[str, Tuple[int, ...]],
        y_gain_inv: Tensor,
        z_gain_inv: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        *y_strings_, z_strings = strings
        assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        hyper_out = self.latent_codec["hyper"].decompress(
            [z_strings], shape["hyper"], z_gain_inv
        )
        y_out = self.latent_codec["y"].decompress(
            y_strings_, shape["y"], hyper_out["params"]
        )
        y_hat = y_out["y_hat"] * y_gain_inv
        return {"y_hat": y_hat}
