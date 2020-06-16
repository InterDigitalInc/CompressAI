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

import pytest

import compressai


def test_get_entropy_coder():
    assert compressai.get_entropy_coder() == 'ans'


def test_available_entropy_coders():
    rv = compressai.available_entropy_coders()

    assert isinstance(rv, list)
    assert 'ans' in rv


def test_set_entropy_coder():
    compressai.set_entropy_coder('ans')

    with pytest.raises(ValueError):
        compressai.set_entropy_coder('cabac')
