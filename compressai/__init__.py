# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from compressai import datasets, entropy_models, layers, models, ops

try:
    from .version import __version__
except ImportError:
    pass

_entropy_coder = 'ans'
_available_entropy_coders = [_entropy_coder]

try:
    import range_coder
    _available_entropy_coders.append('rangecoder')
except ImportError:
    pass


def set_entropy_coder(entropy_coder):
    """
    Specifies the default entropy coder used to encode the bit-streams.

    Use :mod:`available_entropy_coders` to list the possible values.

    Args:
        entropy_coder (string): Name of the entropy coder
    """
    global _entropy_coder  # pylint: disable=W0603
    if entropy_coder not in _available_entropy_coders:
        raise ValueError(
            f'Invalid entropy coder "{entropy_coder}", choose from'
            f'({", ".join(_available_entropy_coders)}).')
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
