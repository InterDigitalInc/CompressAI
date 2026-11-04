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

import argparse
import math
import shutil

from pathlib import Path

import numpy as np
import pytest

from PIL import Image

from compressai.utils.bench.codecs import AV1


@pytest.fixture
def av1_codec():
    encoder = shutil.which("aomenc")
    decoder = shutil.which("aomdec")
    if encoder is None or decoder is None:
        pytest.skip("AV1 integration tests require aomenc and aomdec")
    if Path(encoder).parent != Path(decoder).parent:
        pytest.skip("AV1 binaries must be in the same build directory")
    return AV1(argparse.Namespace(build_dir=str(Path(encoder).parent)))


@pytest.mark.parametrize("mode", ("RGB", "RGBA", "L"))
@pytest.mark.parametrize("path_type", (str, Path))
def test_av1_input_filepath(tmp_path, av1_codec, mode, path_type):
    y, x = np.indices((32, 32))
    rgb = np.stack((x * 8, y * 8, (x + y) * 4), axis=-1).astype(np.uint8)
    image = Image.fromarray(rgb).convert(mode)
    filepath = tmp_path / "input.png"
    image.save(filepath)

    info, rec = av1_codec.run(path_type(filepath), quality=32, return_rec=True)

    assert rec.mode == "RGB"
    assert rec.size == image.size
    assert info["bpp"] > 0
    assert info["encoding_time"] >= 0
    assert info["decoding_time"] >= 0
    assert math.isfinite(info["psnr-rgb"])
