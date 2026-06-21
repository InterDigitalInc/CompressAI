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

"""Pure functional helpers shared by checkerboard latent codecs.

These are extracted from :class:`CheckerboardLatentCodec` so that sibling
codecs (e.g. :class:`MultiContextCheckerboardLatentCodec`) can reuse the
exact same checkerboard split / merge / mask logic without duplicating it.
A single source of truth here also means an anchor-parity boundary fix
applies to every checkerboard codec at once.
"""

from __future__ import annotations

import torch

from torch import Tensor

__all__ = [
    "embed",
    "embed_step",
    "mask_all",
    "mask_all_but_step",
    "merge",
    "step_parity",
    "unembed",
    "write_step",
]


def step_parity(step: str, anchor_parity: str) -> str:
    """Resolve a ``step`` ('anchor' / 'non_anchor') to a parity string."""
    if step == "anchor":
        return anchor_parity
    if step == "non_anchor":
        return "odd" if anchor_parity == "even" else "even"
    raise ValueError(f'Invalid "step" value "{step}"')


def unembed(y: Tensor, *, anchor_parity: str) -> Tensor:
    """Separate single tensor into two even/odd checkerboard chunks.

    .. code-block:: none

        ■ □ ■ □         ■ ■   □ □
        □ ■ □ ■   --->  ■ ■   □ □
        ■ □ ■ □         ■ ■   □ □
    """
    n, c, h, w = y.shape
    y_packed = y.new_zeros((2, n, c, h, w // 2))
    if anchor_parity == "even":
        y_packed[0, ..., 0::2, :] = y[..., 0::2, 0::2]
        y_packed[0, ..., 1::2, :] = y[..., 1::2, 1::2]
        y_packed[1, ..., 0::2, :] = y[..., 0::2, 1::2]
        y_packed[1, ..., 1::2, :] = y[..., 1::2, 0::2]
    else:
        y_packed[0, ..., 0::2, :] = y[..., 0::2, 1::2]
        y_packed[0, ..., 1::2, :] = y[..., 1::2, 0::2]
        y_packed[1, ..., 0::2, :] = y[..., 0::2, 0::2]
        y_packed[1, ..., 1::2, :] = y[..., 1::2, 1::2]
    return y_packed


def embed(y_packed: Tensor, *, anchor_parity: str) -> Tensor:
    """Combine two even/odd checkerboard chunks into single tensor.

    .. code-block:: none

        ■ ■   □ □         ■ □ ■ □
        ■ ■   □ □   --->  □ ■ □ ■
        ■ ■   □ □         ■ □ ■ □
    """
    num_chunks, n, c, h, w_half = y_packed.shape
    assert num_chunks == 2
    y = y_packed.new_zeros((n, c, h, w_half * 2))
    if anchor_parity == "even":
        y[..., 0::2, 0::2] = y_packed[0, ..., 0::2, :]
        y[..., 1::2, 1::2] = y_packed[0, ..., 1::2, :]
        y[..., 0::2, 1::2] = y_packed[1, ..., 0::2, :]
        y[..., 1::2, 0::2] = y_packed[1, ..., 1::2, :]
    else:
        y[..., 0::2, 1::2] = y_packed[0, ..., 0::2, :]
        y[..., 1::2, 0::2] = y_packed[0, ..., 1::2, :]
        y[..., 0::2, 0::2] = y_packed[1, ..., 0::2, :]
        y[..., 1::2, 1::2] = y_packed[1, ..., 1::2, :]
    return y


def embed_step(
    step_index: int, y_i: Tensor, width: int, *, anchor_parity: str
) -> Tensor:
    """Embed a per-step half-width tensor back into a full-grid tensor."""
    n, c, h, _ = y_i.shape
    y_packed = y_i.new_zeros((2, n, c, h, width // 2))
    y_packed[step_index] = y_i
    return embed(y_packed, anchor_parity=anchor_parity)


def write_step(dest: Tensor, src: Tensor, step: str, *, anchor_parity: str) -> None:
    """Copy ``src`` pixels at the current step's positions into ``dest`` in-place."""
    parity = step_parity(step, anchor_parity)
    if parity == "even":
        dest[..., 0::2, 0::2] = src[..., 0::2, 0::2]
        dest[..., 1::2, 1::2] = src[..., 1::2, 1::2]
    else:
        dest[..., 0::2, 1::2] = src[..., 0::2, 1::2]
        dest[..., 1::2, 0::2] = src[..., 1::2, 0::2]


def mask_all_but_step(y: Tensor, step: str, *, anchor_parity: str) -> Tensor:
    """Keep only pixels in the current step, and zero out the rest."""
    y = y.clone()
    parity = step_parity(step, anchor_parity)
    if parity == "even":
        y[..., 0::2, 1::2] = 0
        y[..., 1::2, 0::2] = 0
    else:
        y[..., 0::2, 0::2] = 0
        y[..., 1::2, 1::2] = 0
    return y


def mask_all(y: Tensor) -> Tensor:
    """Return a zero tensor with the same shape, dtype and device as ``y``."""
    return torch.zeros_like(y)


def merge(*args: Tensor) -> Tensor:
    """Concatenate tensors along the channel dimension."""
    return torch.cat(args, dim=1)
