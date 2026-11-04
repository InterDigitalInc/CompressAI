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
import os
import shutil
import subprocess
import tempfile

from pathlib import Path

import numpy as np
import pytest

from PIL import Image

from compressai.utils.bench.codecs import JPEG2000


@pytest.fixture
def jpeg2000_codec():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("JPEG2000 integration tests require ffmpeg")
    encoders = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], text=True)
    if "libopenjpeg" not in encoders:
        pytest.skip("JPEG2000 integration tests require libopenjpeg")
    return JPEG2000(argparse.Namespace())


@pytest.fixture
def codec_tmpdir(tmp_path, monkeypatch):
    directory = tmp_path / "codec-temp"
    directory.mkdir()
    # Redirect only the real temporary-file directory, not the codec subprocess.
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


def _temporary_fds(directory):
    fd_directory = Path("/proc/self/fd")
    if not fd_directory.is_dir():
        return None
    descriptors = set()
    for path in fd_directory.iterdir():
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue
        if target.startswith(str(directory) + os.sep):
            descriptors.add(path.name)
    return descriptors


def test_binary_codec_cleans_up_after_encoder_failure(
    tmp_path, codec_tmpdir, jpeg2000_codec
):
    filepath = tmp_path / "small.png"
    Image.new("RGB", (1, 1), (60, 120, 180)).save(filepath)

    original_fds = _temporary_fds(codec_tmpdir)

    # OpenJPEG's default six resolution levels cannot encode this tiny image.
    for _ in range(2):
        with pytest.raises(SystemExit) as error:
            jpeg2000_codec.run(str(filepath), quality=32)
        assert error.value.code == 1
        assert _temporary_fds(codec_tmpdir) == original_fds
        assert list(codec_tmpdir.iterdir()) == []


def test_binary_codec_cleans_up_after_success(tmp_path, codec_tmpdir, jpeg2000_codec):
    y, x = np.indices((64, 64))
    image = np.stack((x * 4, y * 4, (x + y) * 2), axis=-1).astype(np.uint8)
    filepath = tmp_path / "image.png"
    Image.fromarray(image).save(filepath)

    original_fds = _temporary_fds(codec_tmpdir)
    info, rec = jpeg2000_codec.run(str(filepath), quality=32, return_rec=True)

    assert rec.mode == "RGB"
    assert rec.size == (64, 64)
    assert info["bpp"] > 0
    assert info["encoding_time"] >= 0
    assert info["decoding_time"] >= 0
    assert math.isfinite(info["psnr-rgb"])
    assert _temporary_fds(codec_tmpdir) == original_fds
    assert list(codec_tmpdir.iterdir()) == []
