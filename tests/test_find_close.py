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

import importlib
import os
import re

import pytest

find_close = importlib.import_module("compressai.utils.find_close.__main__")

# Example: GENERATE_EXPECTED=1 pytest -sx tests/test_find_close.py
GENERATE_EXPECTED = os.getenv("GENERATE_EXPECTED")


@pytest.mark.parametrize("codec", ("jpeg",))
@pytest.mark.parametrize(
    "metric, target",
    (
        ("psnr", "30"),
        ("bpp", "0.2"),
    ),
)
def test_find_close(capsys, codec, metric, target):
    here = os.path.dirname(__file__)
    dirpath = os.path.join(here, "assets/dataset/image")

    cmd = [
        codec,
        os.path.join(dirpath, "stmalo_fracape.png"),
        target,
        "-m",
        metric,
    ]

    find_close.main(cmd)

    output = capsys.readouterr().out
    print(output)
    match = next(re.finditer(rf"{metric}:\s([0-9]+\.[0-9]+)", output))

    value = float(match.groups()[0])
    target = float(target)

    tol = 0.05 + 0.05 * abs(value)
    assert abs(value - target) < tol
