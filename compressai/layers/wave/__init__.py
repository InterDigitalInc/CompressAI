"""Generic ``pytorch_wavelets``-backed 2D DWT / IDWT primitives.

Wraps the optional ``pytorch_wavelets`` dependency into a thin
:class:`DWT2D` / :class:`IDWT2D` channel-concatenated interface that
fits naturally into stride-2 conv chains. The dependency is loaded
lazily (``import compressai`` / ``compressai.zoo`` /
``compressai.layers`` stay free of the wavelet stack); construct
:class:`DWT2D` / :class:`IDWT2D` to trigger it.

The AuxT-specific :class:`compressai.models._helpers.auxt.WLS` /
:class:`~compressai.models._helpers.auxt.iWLS` blocks (Li et al., ICLR
2025) are built on top of these wrappers but live alongside their
model-integration helpers in :mod:`compressai.models._helpers.auxt`.
Install the optional extras with ``pip install compressai[wavelet]``.
"""

from __future__ import annotations

from .wavelet import (
    DWT2D,
    DWT_2D,
    IDWT2D,
    IDWT_2D,
    is_pytorch_wavelets_available,
)

__all__ = [
    "DWT2D",
    "DWT_2D",
    "IDWT2D",
    "IDWT_2D",
    "is_pytorch_wavelets_available",
]
