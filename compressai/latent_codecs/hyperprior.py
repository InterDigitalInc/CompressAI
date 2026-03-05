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

from .base import LatentCodec

__all__ = [
    "HyperpriorLatentCodec",
]


@register_module("HyperpriorLatentCodec")
class HyperpriorLatentCodec(LatentCodec):
    """Hyperprior codec constructed from latent codec for ``y`` that
    compresses ``y`` using ``params`` derived from ``z_hat``.

    Hyperprior entropy modeling introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. code-block:: none

                ┌───┐  z  ┌──────┐ z_hat ┌───┐
            ┌─►─┤h_a├──►──┤ lc_z ├───►───┤h_s├─►─┐
            │   └───┘     └──────┘       └───┘   │
            │                                    ▼ params
            │                                    │
            │                                 ┌──┴───┐
        y ──┴───────────────────►─────────────┤ lc_y ├───►── y_hat
                                              └──────┘

    The original hyperprior is the combination of an
    entropy bottleneck for ``z`` and a gaussian conditional for ``y``:

    .. code-block:: none

                 ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            ┌─►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►─┐
            │    └───┘     └───┘        EB        └───┘    │
            │                                              │
            │                  ┌──────────────◄────────────┘
            │                  │            params
            │               ┌──┴──┐
            │               │  EP │
            │               └──┬──┘
            │                  │
            │   ┌───┐  y_hat   ▼
        y ──┴─►─┤ Q ├────►────····────►── y_hat
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

    def forward(self, y: Tensor) -> Dict[str, Any]:
        z = self.h_a(y)
        z_out = self.latent_codec["z"](z)
        z_hat = z_out["y_hat"]
        params = self.h_s(z_hat)
        y_out = self.latent_codec["y"](y, params)
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": z_out["likelihoods"]["y"],
            },
            "y_hat": y_out["y_hat"],
        }

    def compress(self, y: Tensor) -> Dict[str, Any]:
        z = self.h_a(y)
        z_out = self.latent_codec["z"].compress(z)
        z_hat = z_out["y_hat"]
        params = self.h_s(z_hat)
        y_out = self.latent_codec["y"].compress(y, params)
        [z_strings] = z_out["strings"]
        return {
            "strings": [*y_out["strings"], z_strings],
            "shape": {"y": y_out["shape"], "z": z_out["shape"]},
            "y_hat": y_out["y_hat"],
        }

    def decompress(
        self, strings: List[List[bytes]], shape: Dict[str, Tuple[int, ...]], **kwargs
    ) -> Dict[str, Any]:
        *y_strings_, z_strings = strings
        assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        z_out = self.latent_codec["z"].decompress([z_strings], shape["z"])
        z_hat = z_out["y_hat"]
        params = self.h_s(z_hat)
        y_out = self.latent_codec["y"].decompress(y_strings_, shape["y"], params)
        return {"y_hat": y_out["y_hat"]}
