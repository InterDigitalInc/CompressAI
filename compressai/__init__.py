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

from compressai import (
    datasets,
    entropy_models,
    latent_codecs,
    layers,
    losses,
    models,
    ops,
    optimizers,
    registry,
    transforms,
    typing,
    zoo,
)

try:
    from .version import __version__
except ImportError:
    pass

_entropy_coder = "ans"
_available_entropy_coders = [_entropy_coder]

try:
    import range_coder

    _available_entropy_coders.append("rangecoder")
except ImportError:
    pass


def set_entropy_coder(entropy_coder):
    """
    Specifies the default entropy coder used to encode the bit-streams.

    Use :mod:`available_entropy_coders` to list the possible values.

    Args:
        entropy_coder (string): Name of the entropy coder
    """
    global _entropy_coder
    if entropy_coder not in _available_entropy_coders:
        raise ValueError(
            f'Invalid entropy coder "{entropy_coder}", choose from'
            f'({", ".join(_available_entropy_coders)}).'
        )
    _entropy_coder = entropy_coder


def get_entropy_coder():
    """
    Return the name of the default entropy coder used to encode the bit-streams.
    """
    return _entropy_coder


def available_entropy_coders():
    """
    Return the list of available entropy coders.
    """
    return _available_entropy_coders


__all__ = [
    "datasets",
    "entropy_models",
    "latent_codecs",
    "layers",
    "losses",
    "models",
    "ops",
    "optimizers",
    "registry",
    "transforms",
    "typing",
    "zoo",
    "available_entropy_coders",
    "get_entropy_coder",
    "set_entropy_coder",
]
