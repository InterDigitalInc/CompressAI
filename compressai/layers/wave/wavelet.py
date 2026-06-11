"""``pytorch_wavelets``-backed 2D DWT / IDWT wrappers.

Two thin wrappers (:class:`DWT2D` / :class:`IDWT2D`) smooth over
``pytorch_wavelets``' four-tuple subband layout into a single
channel-concatenated tensor that fits naturally into stride-2 conv
chains.

``pytorch_wavelets`` is an optional dependency installed via the
``compressai[wavelet]`` extras. Module import is non-fatal: top-level
import succeeds without it, but constructing :class:`DWT2D` /
:class:`IDWT2D` raises a friendly :class:`ModuleNotFoundError` when the
extras are missing.

The AuxT-specific :class:`compressai.models._helpers.auxt.WLS` /
:class:`~compressai.models._helpers.auxt.iWLS` blocks are built on top
of these wrappers but live alongside their model-integration helpers in
:mod:`compressai.models._helpers.auxt` rather than here, so this module
stays a generic wavelet primitive that future non-AuxT models (e.g.
WeConvene) can also reuse.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ModuleNotFoundError as error:
    DWTForward = None  # type: ignore[assignment]
    DWTInverse = None  # type: ignore[assignment]
    _PYTORCH_WAVELETS_IMPORT_ERROR = error
else:
    _PYTORCH_WAVELETS_IMPORT_ERROR = None

__all__ = [
    "DWT2D",
    "IDWT2D",
    "DWT_2D",
    "IDWT_2D",
    "is_pytorch_wavelets_available",
]


def is_pytorch_wavelets_available() -> bool:
    """Return ``True`` when the optional ``pytorch_wavelets`` package is importable."""
    return DWTForward is not None and DWTInverse is not None


def _require_pytorch_wavelets() -> None:
    if is_pytorch_wavelets_available():
        return
    raise ModuleNotFoundError(
        "Wavelet layers require the optional dependency `pytorch_wavelets`. "
        "Install it via `pip install compressai[wavelet]`."
    ) from _PYTORCH_WAVELETS_IMPORT_ERROR


class DWT2D(nn.Module):
    """Single-level DWT wrapper that channel-concatenates the four subbands.

    Output channels = ``4 * input_channels`` (low + 3 high-pass), spatial
    size halved. Use ``wave="haar"`` for the AuxT defaults.
    """

    def __init__(self, wave: str = "haar", mode: str = "zero") -> None:
        super().__init__()
        _require_pytorch_wavelets()
        self.transform = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, input_tensor: Tensor) -> Tensor:
        lowpass, highpass_pyramid = self.transform(input_tensor)
        [highpass] = highpass_pyramid
        subbands = (
            lowpass,
            highpass[:, :, 0, ...],
            highpass[:, :, 1, ...],
            highpass[:, :, 2, ...],
        )
        return torch.cat(subbands, dim=1)


class IDWT2D(nn.Module):
    """Inverse counterpart of :class:`DWT2D` matching its channel layout."""

    def __init__(self, wave: str = "haar", mode: str = "zero") -> None:
        super().__init__()
        _require_pytorch_wavelets()
        self.inverse = DWTInverse(wave=wave, mode=mode)

    def forward(self, input_tensor: Tensor) -> Tensor:
        lowpass, band_lh, band_hl, band_hh = input_tensor.chunk(4, dim=1)
        highpass = torch.stack((band_lh, band_hl, band_hh), dim=2)
        return self.inverse((lowpass, [highpass]))


# Aliases kept for parity with the upstream AuxT release.
DWT_2D = DWT2D
IDWT_2D = IDWT2D
