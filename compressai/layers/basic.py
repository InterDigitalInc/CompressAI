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

import torch
import torch.nn as nn

from torch import Tensor

__all__ = [
    "Lambda",
    "NamedLayer",
    "Reshape",
    "Transpose",
    "Interleave",
    "Gain",
]


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def __repr__(self):
        return f"{self.__class__.__name__}(func={self.func})"

    def forward(self, x):
        return self.func(x)


class NamedLayer(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def forward(self, x):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def forward(self, x):
        output_shape = (x.shape[0], *self.shape)
        try:
            return x.reshape(output_shape)
        except RuntimeError as e:
            e.args += (f"Cannot reshape input {tuple(x.shape)} to {output_shape}",)
            raise e


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def __repr__(self):
        return f"{self.__class__.__name__}(dim0={self.dim0}, dim1={self.dim1})"

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1).contiguous()


class Interleave(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        g = self.groups
        n, c, *tail = x.shape
        return x.reshape(n, g, c // g, *tail).transpose(1, 2).reshape(x.shape)


class Gain(nn.Module):
    def __init__(self, shape=None, factor: float = 1.0):
        super().__init__()
        self.factor = factor
        self.gain = nn.Parameter(torch.ones(shape))

    def forward(self, x: Tensor) -> Tensor:
        return self.factor * self.gain * x
