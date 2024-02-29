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

from __future__ import annotations

import torch.nn as nn

from compressai.layers.basic import Gain, Interleave, Reshape

GAIN = 10.0


def conv1d_group_seq(
    num_channels,
    groups=None,
    kernel_size=1,
    enabled=("bn", "act"),
    enabled_final=("bn", "act"),
):
    if groups is None:
        groups = [1] * (len(num_channels) - 1)
    assert len(num_channels) == 0 or len(groups) == len(num_channels) - 1
    xs = []
    for i in range(len(num_channels) - 1):
        is_final = i + 1 == len(num_channels) - 1
        xs.append(
            nn.Conv1d(
                num_channels[i], num_channels[i + 1], kernel_size, groups=groups[i]
            )
        )
        # ChannelShuffle is only required between consecutive group convs.
        if not is_final and groups[i] > 1 and groups[i + 1] > 1:
            xs.append(Interleave(groups[i]))
        if "bn" in enabled and (not is_final or "bn" in enabled_final):
            xs.append(nn.BatchNorm1d(num_channels[i + 1]))
        if "act" in enabled and (not is_final or "act" in enabled_final):
            xs.append(nn.ReLU(inplace=True))
    return nn.Sequential(*xs)


def pointnet_g_a_simple(num_channels, groups=None, gain=GAIN):
    return nn.Sequential(
        *conv1d_group_seq(num_channels, groups),
        nn.AdaptiveMaxPool1d(1),
        Gain((num_channels[-1], 1), gain),
    )


def pointnet_g_s_simple(num_channels, gain=GAIN):
    return nn.Sequential(
        Gain((num_channels[0], 1), 1 / gain),
        *conv1d_group_seq(num_channels, enabled=["act"], enabled_final=[]),
        Reshape((num_channels[-1] // 3, 3)),
    )


def pointnet_classification_backend(num_channels):
    return nn.Sequential(
        *conv1d_group_seq(num_channels[:-1], enabled_final=[]),
        nn.Dropout(0.3),
        nn.Conv1d(num_channels[-2], num_channels[-1], 1),
        Reshape((num_channels[-1],)),
    )
