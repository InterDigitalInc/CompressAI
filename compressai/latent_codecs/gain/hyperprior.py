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

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch.nn as nn

from torch import Tensor

from compressai.registry import register_module

from ..base import LatentCodec

__all__ = [
    "GainHyperpriorLatentCodec",
]


@register_module("GainHyperpriorLatentCodec")
class GainHyperpriorLatentCodec(LatentCodec):
    """Hyperprior codec constructed from latent codec for ``y`` that
    compresses ``y`` using ``params`` derived from ``z_hat``.

    Gain-controlled hyperprior introduced in
    `"Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation"
    <https://arxiv.org/abs/2003.02012>`_, by Ze Cui, Jing Wang,
    Shangyin Gao, Bo Bai, Tiansheng Guo, and Yihui Feng, CVPR, 2021.

    .. code-block:: none

                        z_gain  z_gain_inv
                           │        │
                           ▼        ▼
                ┌───┐  z  ┌┴────────┴┐ z_hat ┌───┐
            ┌─►─┤h_a├──►──┤   lc_z   ├───►───┤h_s├──►─┐
            │   └───┘     └──────────┘       └───┘    │
            │                                         │
            │                y_gain                   ▼ params   y_gain_inv
            │                   │                     │              │
            │                   ▼                     │              ▼
            │                   │                  ┌──┴───┐          │
        y ──┴───────────────►───×───►──────────────┤ lc_y ├────►─────×─────►── y_hat
                                                   └──────┘

    The original gain hyperprior is the combination of an
    entropy bottleneck for ``z`` and a gaussian conditional for ``y``:

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
     - entropy bottleneck ``z`` and gaussian conditional ``y``
     - entropy bottleneck ``z`` and autoregressive ``y``
    """

    def __init__(
        self,
        latent_codec: Mapping[str, LatentCodec],
        h_a: Optional[nn.Module] = None,
        h_s: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.h_a = h_a or nn.Identity()
        self.h_s = h_s or nn.Identity()
        latent_codec = {
            "y": latent_codec["y"],
            "z": latent_codec.get("z") or latent_codec["hyper"],
        }
        self.y = latent_codec["y"]
        self.z = latent_codec["z"]
        self.latent_codec = latent_codec

    def forward(
        self,
        y: Tensor,
        y_gain: Tensor,
        z_gain: Tensor,
        y_gain_inv: Tensor,
        z_gain_inv: Tensor,
    ) -> Dict[str, Any]:
        z = self.h_a(y)
        z = z * z_gain
        z_out = self.latent_codec["z"](z)
        z_hat = z_out["y_hat"]
        z_hat = z_hat * z_gain_inv
        params = self.h_s(z_hat)
        y_out = self.latent_codec["y"](y * y_gain, params)
        y_hat = y_out["y_hat"] * y_gain_inv
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": z_out["likelihoods"]["y"],
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
        z = self.h_a(y)
        z = z * z_gain
        z_out = self.latent_codec["z"].compress(z)
        z_hat = z_out["y_hat"]
        z_hat = z_hat * z_gain_inv
        params = self.h_s(z_hat)
        y_out = self.latent_codec["y"].compress(y * y_gain, params)
        y_hat = y_out["y_hat"] * y_gain_inv
        return {
            "strings": [*y_out["strings"], *z_out["strings"]],
            "shape": {"y": y_out["shape"], "z": z_out["shape"]},
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
        z_out = self.latent_codec["z"].decompress([z_strings], shape["z"])
        z_hat = z_out["y_hat"]
        z_hat = z_hat * z_gain_inv
        params = self.h_s(z_hat)
        y_out = self.latent_codec["y"].decompress(y_strings_, shape["y"], params)
        y_hat = y_out["y_hat"] * y_gain_inv
        return {"y_hat": y_hat}
