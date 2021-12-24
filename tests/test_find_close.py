# Copyright (c) 2021-2022, InterDigital Communications, Inc
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
