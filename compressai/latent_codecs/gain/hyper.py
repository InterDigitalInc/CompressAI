# Copyright (c) 2021-2022, InterDigital Communications, Inc
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

from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.registry import register_module

from ..base import LatentCodec

__all__ = [
    "GainHyperLatentCodec",
]


@register_module("GainHyperLatentCodec")
class GainHyperLatentCodec(LatentCodec):
    """Entropy bottleneck codec with surrounding `h_a` and `h_s` transforms.

    Gain-controlled side branch for hyperprior introduced in
    `"Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation"
    <https://arxiv.org/abs/2003.02012>`_, by Ze Cui, Jing Wang,
    Shangyin Gao, Bo Bai, Tiansheng Guo, and Yihui Feng, CVPR, 2021.

    .. note:: ``GainHyperLatentCodec`` should be used inside
       ``GainHyperpriorLatentCodec`` to construct a full hyperprior.

    .. code-block:: none

                       gain                        gain_inv
                         │                             │
                         ▼                             ▼
               ┌───┐  z  │     ┌───┐ z_hat      z_hat  │       ┌───┐
        y ──►──┤h_a├──►──×──►──┤ Q ├───►───····───►────×────►──┤h_s├──►── params
               └───┘           └───┘        EB                 └───┘

    """

    entropy_bottleneck: EntropyBottleneck
    h_a: nn.Module
    h_s: nn.Module

    def __init__(
        self,
        entropy_bottleneck: Optional[EntropyBottleneck] = None,
        h_a: Optional[nn.Module] = None,
        h_s: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        assert entropy_bottleneck is not None
        self.entropy_bottleneck = entropy_bottleneck
        self.h_a = h_a or nn.Identity()
        self.h_s = h_s or nn.Identity()

    def forward(self, y: Tensor, gain: Tensor, gain_inv: Tensor) -> Dict[str, Any]:
        z = self.h_a(y)
        z = z * gain
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * gain_inv
        params = self.h_s(z_hat)
        return {"likelihoods": {"z": z_likelihoods}, "params": params}

    def compress(self, y: Tensor, gain: Tensor, gain_inv: Tensor) -> Dict[str, Any]:
        z = self.h_a(y)
        z = z * gain
        shape = z.size()[-2:]
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        z_hat = z_hat * gain_inv
        params = self.h_s(z_hat)
        return {"strings": [z_strings], "shape": shape, "params": params}

    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, int], gain_inv: Tensor
    ) -> Dict[str, Any]:
        (z_strings,) = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        z_hat = z_hat * gain_inv
        params = self.h_s(z_hat)
        return {"params": params}
