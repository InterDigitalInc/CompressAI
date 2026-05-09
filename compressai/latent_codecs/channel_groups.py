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

from itertools import accumulate
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.registry import register_module

from .base import LatentCodec

__all__ = [
    "ChannelGroupsLatentCodec",
]


@register_module("ChannelGroupsLatentCodec")
class ChannelGroupsLatentCodec(LatentCodec):
    """Reconstructs groups of channels using previously decoded groups.

    Context model from [Minnen2020] and [He2022].
    Also known as a "channel-conditional" (CC) entropy model.

    See :py:class:`~compressai.models.sensetime.Elic2022Official`
    for example usage.

    [Minnen2020]: `"Channel-wise Autoregressive Entropy Models for
    Learned Image Compression" <https://arxiv.org/abs/2007.08739>`_, by
    David Minnen, and Saurabh Singh, ICIP 2020.

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.
    """

    latent_codec: Mapping[str, LatentCodec]

    channel_context: Mapping[str, nn.Module]

    def __init__(
        self,
        latent_codec: Mapping[str, LatentCodec],
        channel_context: Mapping[str, nn.Module],
        *,
        groups: List[int],
        max_support_slices: int = -1,
        support_filter: Optional[Callable[[int, List[Tensor]], List[Tensor]]] = None,
        side_in_context: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self.groups = list(groups)
        self.groups_acc = list(accumulate(self.groups, initial=0))
        self.max_support_slices = int(max_support_slices)
        self.support_filter = support_filter
        self.side_in_context = bool(side_in_context)
        self.channel_context = nn.ModuleDict(channel_context)
        self.latent_codec = nn.ModuleDict(latent_codec)
        if self.side_in_context and "y0" not in self.channel_context:
            raise ValueError(
                "side_in_context=True requires a channel_context entry for 'y0' "
                "(slice 0's channel_context absorbs side_params instead of "
                "ChannelGroupsLatentCodec returning side_params raw)"
            )

    def forward(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        y_ = torch.split(y, self.groups, dim=1)
        y_out_ = [{}] * len(self.groups)
        y_hat_ = [Tensor()] * len(self.groups)
        y_likelihoods_ = [Tensor()] * len(self.groups)

        for k in range(len(self.groups)):
            params = self._get_ctx_params(k, side_params, y_hat_)
            y_out_[k] = self.latent_codec[f"y{k}"](y_[k], params)
            y_hat_[k] = y_out_[k]["y_hat"]
            y_likelihoods_[k] = y_out_[k]["likelihoods"]["y"]

        y_hat = torch.cat(y_hat_, dim=1)
        y_likelihoods = torch.cat(y_likelihoods_, dim=1)

        return {
            "likelihoods": {
                "y": y_likelihoods,
            },
            "y_hat": y_hat,
        }

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        y_ = torch.split(y, self.groups, dim=1)
        y_out_ = [{}] * len(self.groups)
        y_hat = torch.zeros_like(y)
        y_hat_ = y_hat.split(self.groups, dim=1)

        for k in range(len(self.groups)):
            params = self._get_ctx_params(k, side_params, y_hat_)
            y_out_[k] = self.latent_codec[f"y{k}"].compress(y_[k], params)
            y_hat_[k][:] = y_out_[k]["y_hat"]

        y_strings_groups = [y_out["strings"] for y_out in y_out_]
        assert all(len(y_strings_groups[0]) == len(ss) for ss in y_strings_groups)

        return {
            "strings": [s for ss in y_strings_groups for s in ss],
            "shape": [y_out["shape"] for y_out in y_out_],
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: List[Tuple[int, ...]],
        side_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        n = len(strings[0])
        assert all(len(ss) == n for ss in strings)
        strings_per_group = len(strings) // len(self.groups)

        y_out_ = [{}] * len(self.groups)
        y_shape = (sum(s[0] for s in shape), *shape[0][1:])
        y_hat = torch.zeros((n, *y_shape), device=side_params.device)
        y_hat_ = y_hat.split(self.groups, dim=1)

        for k in range(len(self.groups)):
            params = self._get_ctx_params(k, side_params, y_hat_)
            y_out_[k] = self.latent_codec[f"y{k}"].decompress(
                strings[strings_per_group * k : strings_per_group * (k + 1)],
                shape[k],
                params,
            )
            y_hat_[k][:] = y_out_[k]["y_hat"]

        return {
            "y_hat": y_hat,
        }

    def merge_y(self, *args):
        return torch.cat(args, dim=1)

    def merge_params(self, *args):
        return torch.cat(args, dim=1)

    def _get_ctx_params(
        self, k: int, side_params: Tensor, y_hat_: List[Tensor]
    ) -> Tensor:
        if self.side_in_context:
            return self._get_ctx_params_side_in_context(k, side_params, y_hat_)
        if k == 0:
            return side_params
        support = self._select_support(k, y_hat_)
        ch_ctx_params = self.channel_context[f"y{k}"](self.merge_y(*support))
        return self.merge_params(ch_ctx_params, side_params)

    def _get_ctx_params_side_in_context(
        self, k: int, side_params: Tensor, y_hat_: List[Tensor]
    ) -> Tensor:
        # Family 1 layout (STF / WACNN / TCM / CCA): ``channel_context.y{k}``
        # absorbs ``side_params`` directly so its mean_cc / scale_cc heads can
        # see the hyperprior latent_means / latent_scales alongside the
        # previously decoded slices. The head's output already encodes the
        # final per-slice (mean, scale) prediction, so no further cat with
        # ``side_params`` is needed before the leaf.
        if k == 0:
            return self.channel_context["y0"](side_params)
        support = self._select_support(k, y_hat_)
        ch_input = self.merge_params(side_params, self.merge_y(*support))
        return self.channel_context[f"y{k}"](ch_input)

    def _select_support(self, k: int, y_hat_: List[Tensor]) -> List[Tensor]:
        prior = list(y_hat_[:k])
        if self.support_filter is not None:
            return list(self.support_filter(k, prior))
        if self.max_support_slices < 0:
            return prior
        return prior[: self.max_support_slices]
