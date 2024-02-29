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

from math import ceil

import torch

try:
    from pointops.functions import pointops
except ImportError:
    pass  # NOTE: Optional dependency.


def index_points(xyzs, idx):
    """Index points.

    Args:
        xyzs: (b, c, n)
        idx: (b, ...)

    Returns:
        xyzs_out: (b, c, ...)
    """
    _, c, _ = xyzs.shape
    b, *idx_dims = idx.shape
    idx_out = idx.reshape(b, 1, -1).repeat(1, c, 1)
    xyzs_out = xyzs.gather(2, idx_out).reshape(b, c, *idx_dims)
    return xyzs_out


def select_xyzs_and_feats(
    candidate_xyzs,
    candidate_feats,
    upsample_num,
    upsample_rate=None,
    method="batch_loop",
):
    """Selects subset of points to match predicted local point cloud densities.

    Args:
        candidate_xyzs: (b, 3, n, s)
        candidate_feats: (b, c, n, s)
        upsample_num: (b, n)
        upsample_rate: Maximum number of points per group.
        method: "batch_loop" or "batch_noloop".

    Returns:
        xyzs: (b, 3, m)
        feats: (b, c, m)
    """
    device = candidate_xyzs.device
    b, c, n, s = candidate_feats.shape

    # Prefer faster method, if applicable.
    if b == 1 and upsample_rate is None:
        xyzs, feats = _select_xyzs_and_feats_single(
            candidate_xyzs, candidate_feats, upsample_num
        )
        return xyzs, feats

    # Select upsample_num points, then resample to a fixed number of points.
    if method == "batch_loop":
        max_points = ceil(n * upsample_rate)
        xyzs = []
        feats = []

        for i in range(b):
            xyzs_i, feats_i = _select_xyzs_and_feats_single(
                candidate_xyzs[[i]], candidate_feats[[i]], upsample_num[[i]]
            )
            xyzs_i, feats_i = resample_points(xyzs_i, feats_i, max_points)
            xyzs.append(xyzs_i)
            feats.append(feats_i)

        xyzs = torch.cat(xyzs)
        feats = torch.cat(feats)

    # Sometimes a bit faster than batch_loop.
    if method == "batch_noloop":
        upsample_num = upsample_num.round().long().clip(1, s)

        # Select upsample_num points from each group.
        idx = torch.arange(s, device=device).repeat(b, n, 1)
        mask = idx < upsample_num.unsqueeze(-1)

        # Initialize random permutation.
        idx = randperm((b, n, s), device=device, dim=-1)

        # Flatten point groups.
        idx += torch.arange(n, device=device).view(1, -1, 1) * s
        idx = idx.view(b, -1)
        mask = mask.view(b, n * s)

        # Reorder selected points from various groups to the beginning.
        # TODO(perf): A batch-wise .nonzero() could be faster.
        perm = mask.to(torch.uint8).argsort(dim=-1, descending=True)
        idx = idx.gather(-1, perm)

        # NOTE(perf): Using consistent shapes is much faster on the GPU,
        # so specifying upsample_rate is preferred.
        max_points = (
            mask.sum(dim=-1).max().item()
            if upsample_rate is None
            else ceil(n * upsample_rate)
        )

        # Reduce dimensionality to maximum number of points.
        idx = idx[..., :max_points]

        # Cycle selected points if there are not enough.
        idx, _, _ = cycle_after(idx, mask.sum(dim=-1))

        # Shuffle points (usually not necessary).
        idx = idx.gather(-1, randperm(idx.shape, device=device, dim=-1))

        xyzs = index_points(candidate_xyzs.view(b, 3, -1), idx)
        feats = index_points(candidate_feats.view(b, c, -1), idx)

    return xyzs, feats


def _select_xyzs_and_feats_single(candidate_xyzs, candidate_feats, upsample_num):
    # candidate_xyzs: (b, 3, n, max_upsample_num)
    # candidate_feats: (b, c, n, max_upsample_num)
    # upsample_num: (b, n)
    # mask: (n*max_upsample_num)
    # idx: (b, m)
    # xyzs: (b, 3, m)
    # feats: (b, c, m)

    batch_size, _, points_num, max_upsample_num = candidate_xyzs.shape
    assert batch_size == 1

    # Create mask denoting the first upsample_num points per group:
    upsample_num = upsample_num.round().long().squeeze(0).view(-1, 1)
    mask = torch.arange(max_upsample_num).cuda().view(1, -1).repeat(points_num, 1)
    mask = (mask < upsample_num).view(-1)

    # Convert mask to indices:
    [idx] = mask.nonzero(as_tuple=True)
    idx = idx.unsqueeze(0)

    # Select the first upsample_num xyzs and feats:
    xyzs = index_points(candidate_xyzs.view(*candidate_xyzs.shape[:2], -1), idx)
    feats = index_points(candidate_feats.view(*candidate_feats.shape[:2], -1), idx)

    return xyzs, feats


def resample_points(xyzs, feats, num_points):
    """Resample points to a target number.

    Args:
        xyzs: (b, 3, n)
        feats: (b, c, n)

    Returns:
        new_xyzs: (b, 3, num_points)
        new_feats: (b, c, num_points)
    """
    b, _, n = xyzs.shape
    device = xyzs.device
    assert b == 1

    if n == num_points:
        return xyzs, feats

    # Subsample points if there are too many.
    if n > num_points:
        xyzs_tr = xyzs.permute(0, 2, 1).contiguous()
        idx = pointops.furthestsampling(xyzs_tr, num_points).long()

    # Repeat and create randomly duplicated points if there are not enough.
    elif n < num_points:
        idx_repeated = torch.arange(n, device=device).repeat(num_points // n)
        idx_random = (
            torch.multinomial(torch.ones(n, device=device), num_points % n)
            if num_points % n > 0
            else torch.arange(0, device=device)
        )
        idx = torch.cat((idx_repeated, idx_random))

        # Shuffle; probably unnecessary:
        perm = torch.randperm(len(idx), device=device)
        idx = idx[perm]

    idx = idx.reshape(1, -1)
    new_xyzs = index_points(xyzs, idx)
    new_feats = index_points(feats, idx)

    return new_xyzs, new_feats


def randperm(shape, device=None, dim=-1):
    """Random permutation, like `torch.randperm`, but with a shape."""
    if dim != -1:
        raise NotImplementedError
    idx = torch.rand(shape, device=device).argsort(dim=dim)
    return idx


def cycle_after(x, end):
    """Cycle tensor after a given index.

    Example:

        .. code-block:: python

            >>> x = torch.tensor([[5, 0, 7, 6, 2], [3, 1, 4, 8, 9]])
            >>> end = torch.tensor([2, 3])
            >>> idx, _, _ = cycle_after(x, end)
            >>> idx
            tensor([[5, 0, 5, 0, 5], [3, 1, 4, 3, 1]])
    """
    *dims, n = x.shape
    assert end.shape == tuple(dims)
    idx = torch.arange(n, device=x.device).repeat(*dims, 1)
    mask = idx >= end.unsqueeze(-1)
    idx[mask] %= end.unsqueeze(-1).repeat([1] * len(dims) + [n])[mask]
    x = x.gather(-1, idx)
    return x, idx, ~mask
