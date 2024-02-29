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

import torch.nn as nn

from torch import Tensor

from compressai.layers.basic import Interleave, Reshape, Transpose


class UpsampleBlock(nn.Module):
    def __init__(self, D, E, M, P, S, i, extra_in_ch=3, groups=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                E[i + 1] + (D[i] + extra_in_ch) * bool(M[i]), D[i], 1, groups=groups[0]
            ),
            Interleave(groups=groups[0]),
            nn.BatchNorm1d(D[i]),
            nn.ReLU(inplace=True),
            nn.Conv1d(D[i], E[i] * S[i], 1, groups=groups[1]),
            Interleave(groups=groups[1]),
            nn.BatchNorm1d(E[i] * S[i]),
            nn.ReLU(inplace=True),
            Reshape((E[i], S[i], P[i])),
            Transpose(-2, -1),
            Reshape((E[i], P[i] * S[i])),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
