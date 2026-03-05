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

from typing import Any, Dict, List, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import GaussianConditional
from compressai.layers import MaskedConv2d
from compressai.registry import register_module

from .base import LatentCodec

__all__ = [
    "RasterScanLatentCodec",
]

K = TypeVar("K")
V = TypeVar("V")


@register_module("RasterScanLatentCodec")
class RasterScanLatentCodec(LatentCodec):
    """Autoregression in raster-scan order with local decoded context.

    PixelCNN context model introduced in
    `"Pixel Recurrent Neural Networks"
    <http://arxiv.org/abs/1601.06759>`_,
    by Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu,
    International Conference on Machine Learning (ICML), 2016.

    First applied to learned image compression in
    `"Joint Autoregressive and Hierarchical Priors for Learned Image
    Compression" <https://arxiv.org/abs/1809.02736>`_,
    by D. Minnen, J. Balle, and G.D. Toderici,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                         ctx_params
                             │
                             ▼
                             │ ┌───◄───┐
                           ┌─┴─┴─┐  ┌──┴──┐
                           │  EP │  │  CP │
                           └──┬──┘  └──┬──┘
                              │        │
                              │        ▲
               ┌───┐  y_hat   ▼        │
        y ──►──┤ Q ├────►────····───►──┴──►── y_hat
               └───┘          GC

    """

    def __init__(
        self,
        gaussian_conditional: GaussianConditional,
        entropy_parameters: nn.Module,
        context_prediction: MaskedConv2d,
        **kwargs,
    ):
        super().__init__()
        self.gaussian_conditional = gaussian_conditional
        self.entropy_parameters = entropy_parameters
        self.context_prediction = context_prediction
        [k, *ks] = self.context_prediction.kernel_size
        assert all(k == k_ for k_ in ks)
        self.kernel_size = k
        self.padding = (self.kernel_size - 1) // 2

    def forward(self, y: Tensor, params: Tensor) -> Dict[str, Any]:
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.merge(params, self.context_prediction(y_hat))
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        return {"likelihoods": {"y": y_likelihoods}, "y_hat": y_hat}

    def compress(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        n, _, y_height, y_width = y.shape
        ds = []
        for i in range(n):
            encoder = BufferedRansEncoder()
            y_hat = self._compress_single_stream(
                encoder=encoder,
                y=y[i : i + 1, :, :, :],
                params=ctx_params[i : i + 1, :, :, :],
                height=y_height,
                width=y_width,
            )
            y_strings = encoder.flush()
            ds.append({"strings": [y_strings], "y_hat": y_hat.squeeze(0)})
        return {**default_collate(ds), "shape": y.shape[2:4]}

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, int],
        ctx_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        (y_strings,) = strings
        y_height, y_width = shape
        ds = []
        for i in range(len(y_strings)):
            decoder = RansDecoder()
            decoder.set_stream(y_strings[i])
            y_hat = self._decompress_single_stream(
                decoder=decoder,
                params=ctx_params[i : i + 1, :, :, :],
                height=y_height,
                width=y_width,
                device=ctx_params.device,
            )
            ds.append({"y_hat": y_hat.squeeze(0)})
        return default_collate(ds)

    def _compress_single_stream(
        self,
        encoder: BufferedRansEncoder,
        y: Tensor,
        params: Tensor,
        *,
        height: int,
        width: int,
    ) -> Tensor:
        """Compresses y and writes to encoder bitstream.

        Returns:
            The y_hat that will be reconstructed at the decoder.
        """
        assert height == y.shape[-2]
        assert width == y.shape[-1]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        masked_weight = self.context_prediction.weight * self.context_prediction.mask

        y_hat = _pad_2d(y, self.padding)

        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        for h in range(height):
            for w in range(width):
                # only perform the mask convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + self.kernel_size, w : w + self.kernel_size]
                ctx_p = F.conv2d(y_crop, masked_weight, self.context_prediction.bias)

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(self.merge(p, ctx_p))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, self.padding, self.padding]
                symbols = self.gaussian_conditional.quantize(
                    y_crop, "symbols", means_hat
                )
                y_hat_item = symbols + means_hat

                hp = h + self.padding
                wp = w + self.padding
                y_hat[:, :, hp, wp] = y_hat_item

                symbols_list.extend(symbols.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        y_hat = _pad_2d(y_hat, -self.padding)
        return y_hat

    def _decompress_single_stream(
        self,
        decoder: RansDecoder,
        params: Tensor,
        *,
        height: int,
        width: int,
        device,
    ) -> Tensor:
        """Decodes y_hat from decoder bitstream.

        Returns:
            The reconstructed y_hat.
        """
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        masked_weight = self.context_prediction.weight * self.context_prediction.mask

        c = self.context_prediction.in_channels
        shape = (1, c, height + 2 * self.padding, width + 2 * self.padding)
        y_hat = torch.zeros(shape, device=device)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the mask convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + self.kernel_size, w : w + self.kernel_size]
                ctx_p = F.conv2d(y_crop, masked_weight, self.context_prediction.bias)

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(self.merge(p, ctx_p))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                symbols = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                symbols = Tensor(symbols).reshape(1, -1)
                y_hat_item = self.gaussian_conditional.dequantize(symbols, means_hat)

                hp = h + self.padding
                wp = w + self.padding
                y_hat[:, :, hp, wp] = y_hat_item

        y_hat = _pad_2d(y_hat, -self.padding)
        return y_hat

    def merge(self, *args):
        return torch.cat(args, dim=1)


def _pad_2d(x: Tensor, padding: int) -> Tensor:
    return F.pad(x, (padding, padding, padding, padding))


def default_collate(batch: List[Dict[K, V]]) -> Dict[K, List[V]]:
    """Combines a list of dictionaries into a single dictionary.

    Workaround to ``torch.utils.data.default_collate`` bug in PyTorch 2.0.0.
    """
    if not isinstance(batch, list) or any(not isinstance(d, dict) for d in batch):
        raise NotImplementedError

    result = _ld_to_dl(batch)

    for k, vs in result.items():
        if all(isinstance(v, Tensor) for v in vs):
            result[k] = torch.stack(vs)

    return result


def _ld_to_dl(ld: List[Dict[K, V]]) -> Dict[K, List[V]]:
    dl = {}
    for d in ld:
        for k, v in d.items():
            if k not in dl:
                dl[k] = []
            dl[k].append(v)
    return dl
