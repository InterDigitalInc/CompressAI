"""Slice-conditional entropy backbone shared by WACNN / SymmetricalTransFormer / MambaVC.

Promoted out of the historical ``models/stf_support.py`` so the abstract base
class is discoverable by name. Channel-counting helpers and a parameterised
entropy-transform factory live here too — they used to be duplicated across
``stf_support`` / ``ssm_support`` / ``weconvene_support``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import ChannelSliceLatentCodec
from compressai.models.utils import conv

from ..base import CompressionModel

__all__ = [
    "SliceEntropyCompressionModel",
    "infer_max_support_slices",
    "infer_num_slices",
    "lrp_support_channels",
    "make_entropy_transform",
    "slice_support_channels",
]


_DEFAULT_NUM_SLICES_PREFIX = "latent_codec.cc_mean_transforms."
_KEY_SUFFIX = ".0.weight"


def slice_support_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * index
    return latent_channels + slice_channels * min(index, max_support_slices)


def lrp_support_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * (index + 1)
    return latent_channels + slice_channels * min(index + 1, max_support_slices + 1)


def make_entropy_transform(
    in_channels: int,
    out_channels: int,
    *,
    widths: Sequence[int] = (224, 128),
) -> nn.Sequential:
    """Stack of stride-1 3x3 convs with GELU between, used by every slice
    entropy model. ``widths`` specifies hidden conv widths; defaults to the
    Mamba/WeConvene 3-conv stack. Pass ``widths=(224, 176, 128, 64)`` for the
    STF/WACNN 5-conv stack."""
    layers: list[nn.Module] = []
    prev = in_channels
    for width in widths:
        layers.append(conv(prev, width, stride=1, kernel_size=3))
        layers.append(nn.GELU())
        prev = width
    layers.append(conv(prev, out_channels, stride=1, kernel_size=3))
    return nn.Sequential(*layers)


def infer_num_slices(
    state_dict: Dict[str, Tensor],
    *,
    prefix: str = _DEFAULT_NUM_SLICES_PREFIX,
    suffix: str = _KEY_SUFFIX,
) -> int:
    slice_indices = {
        int(key[len(prefix) :].split(".", 1)[0])
        for key in state_dict
        if key.startswith(prefix) and key.endswith(suffix)
    }
    return len(slice_indices)


def infer_max_support_slices(
    state_dict: Dict[str, Tensor],
    latent_channels: int,
    num_slices: int,
    *,
    prefix: str = _DEFAULT_NUM_SLICES_PREFIX,
    suffix: str = _KEY_SUFFIX,
    extra_factor: int = 1,
) -> int:
    """Infer ``max_support_slices`` from the input width of the first
    cc_mean transform conv. ``extra_factor`` accounts for models like DCAE/SAAF
    that prepend additional copies of the latent (``M*3 + slice_channels*N``);
    pass ``extra_factor=3`` there. Slice-only models (STF/Mamba*) keep the
    default ``extra_factor=1``."""
    slice_channels = latent_channels // num_slices
    matching = [
        tensor.size(1)
        for key, tensor in state_dict.items()
        if key.startswith(prefix) and key.endswith(suffix)
    ]
    if not matching:
        return 0
    max_input_channels = max(matching)
    return max(
        0, (max_input_channels - extra_factor * latent_channels) // slice_channels
    )


class SliceEntropyCompressionModel(CompressionModel):
    """Channel-conditional entropy backbone shared by WACNN, SymmetricalTransFormer, MambaVC.

    Subclasses must populate ``g_a``, ``g_s``, ``h_a``, ``h_mean_s`` and
    ``h_scale_s``, then call :meth:`_init_slice_entropy` to wire up the
    entropy bottleneck for ``z`` and the :class:`ChannelSliceLatentCodec`
    for ``y``.
    """

    h_a: nn.Module
    h_mean_s: nn.Module
    h_scale_s: nn.Module
    entropy_bottleneck: EntropyBottleneck
    latent_codec: ChannelSliceLatentCodec

    def _init_slice_entropy(
        self,
        latent_channels: int,
        entropy_bottleneck_channels: int,
        num_slices: int,
        max_support_slices: int,
        mean_support_transforms: Optional[nn.ModuleList] = None,
        scale_support_transforms: Optional[nn.ModuleList] = None,
    ) -> None:
        if latent_channels % num_slices != 0:
            raise ValueError("latent_channels must be divisible by num_slices")
        if (
            mean_support_transforms is not None
            and len(mean_support_transforms) != num_slices
        ):
            raise ValueError("mean_support_transforms must have num_slices entries")
        if (
            scale_support_transforms is not None
            and len(scale_support_transforms) != num_slices
        ):
            raise ValueError("scale_support_transforms must have num_slices entries")

        slice_channels = latent_channels // num_slices
        widths = (224, 176, 128, 64)
        cc_mean_transforms = nn.ModuleList(
            make_entropy_transform(
                slice_support_channels(
                    latent_channels, slice_channels, index, max_support_slices
                ),
                slice_channels,
                widths=widths,
            )
            for index in range(num_slices)
        )
        cc_scale_transforms = nn.ModuleList(
            make_entropy_transform(
                slice_support_channels(
                    latent_channels, slice_channels, index, max_support_slices
                ),
                slice_channels,
                widths=widths,
            )
            for index in range(num_slices)
        )
        lrp_transforms = nn.ModuleList(
            make_entropy_transform(
                lrp_support_channels(
                    latent_channels, slice_channels, index, max_support_slices
                ),
                slice_channels,
                widths=widths,
            )
            for index in range(num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        self.latent_codec = ChannelSliceLatentCodec(
            cc_mean_transforms=cc_mean_transforms,
            cc_scale_transforms=cc_scale_transforms,
            lrp_transforms=lrp_transforms,
            mean_support_transforms=mean_support_transforms,
            scale_support_transforms=scale_support_transforms,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            quantizer="ste",
        )

    @property
    def num_slices(self) -> int:
        return self.latent_codec.num_slices

    @property
    def max_support_slices(self) -> int:
        return self.latent_codec.max_support_slices

    def _hyper_priors(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        return z, z_likelihoods, latent_means, latent_scales

    def _forward_latent_output(
        self, y: Tensor
    ) -> Dict[str, Dict[str, Tensor] | Tensor]:
        _, z_likelihoods, latent_means, latent_scales = self._hyper_priors(y)
        y_out = self.latent_codec(y, latent_means, latent_scales)
        output: Dict[str, Dict[str, Tensor] | Tensor] = {
            "y_hat": y_out["y_hat"],
            "likelihoods": {"y": y_out["likelihoods"]["y"], "z": z_likelihoods},
        }
        return output

    def _forward_latent(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        output = self._forward_latent_output(y)
        return output["y_hat"], output["likelihoods"]["y"], output["likelihoods"]["z"]

    def _compress_latent(self, y: Tensor) -> Dict[str, object]:
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_out = self.latent_codec.compress(y, latent_means, latent_scales)
        return {
            "strings": [[y_out["strings"][0]], z_strings],
            "shape": z.size()[-2:],
        }

    def _decompress_latent(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Tensor:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_shape = (z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        y_out = self.latent_codec.decompress(
            strings[0], y_shape, latent_means, latent_scales
        )
        return y_out["y_hat"]
