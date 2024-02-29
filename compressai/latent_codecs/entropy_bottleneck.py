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

from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.registry import register_module

from .base import LatentCodec

__all__ = [
    "EntropyBottleneckLatentCodec",
]


@register_module("EntropyBottleneckLatentCodec")
class EntropyBottleneckLatentCodec(LatentCodec):
    """Entropy bottleneck codec.

    Factorized prior "entropy bottleneck" introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. code-block:: none

               ┌───┐ y_hat
        y ──►──┤ Q ├───►───····───►─── y_hat
               └───┘        EB

    """

    entropy_bottleneck: EntropyBottleneck

    def __init__(
        self,
        entropy_bottleneck: Optional[EntropyBottleneck] = None,
        **kwargs,
    ):
        super().__init__()
        self.entropy_bottleneck = entropy_bottleneck or EntropyBottleneck(**kwargs)

    def forward(self, y: Tensor) -> Dict[str, Any]:
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return {"likelihoods": {"y": y_likelihoods}, "y_hat": y_hat}

    def compress(self, y: Tensor) -> Dict[str, Any]:
        shape = y.size()[-2:]
        y_strings = self.entropy_bottleneck.compress(y)
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        return {"strings": [y_strings], "shape": shape, "y_hat": y_hat}

    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, int], **kwargs
    ) -> Dict[str, Any]:
        (y_strings,) = strings
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        return {"y_hat": y_hat}
