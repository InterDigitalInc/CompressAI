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
import json
import os

from pathlib import Path

import numpy as np
import pytest

bench_codec = importlib.import_module("compressai.utils.video.bench.__main__")

# Example: GENERATE_EXPECTED=1 pytest -sx tests/test_bench_codec_video.py
GENERATE_EXPECTED = os.getenv("GENERATE_EXPECTED")


def test_bench_codec_cmd():
    with pytest.raises(SystemExit):
        bench_codec.main([])

    with pytest.raises(SystemExit):
        bench_codec.main(["x264"])


@pytest.mark.skip(reason="test requires ffmpeg")
@pytest.mark.parametrize("codec", ("x264", "x265"))  # , "VTM", "HM"))
def test_bench_codec_video(capsys, codec, tmp_path):
    here = Path(__file__).parent
    input_dir_path = here / "assets/dataset/video"

    cmd = [codec, input_dir_path.as_posix(), tmp_path.as_posix()]

    bench_codec.main(cmd)

    output = capsys.readouterr().out
    output = json.loads(output)
    expected = here / f"expected/bench_{codec}.json"

    if not expected.is_file():
        if not GENERATE_EXPECTED:
            raise RuntimeError(f"Missing expected file {expected}")
        with open(expected, "w") as f:
            json.dump(output, f)

    with open(expected, "r") as f:
        expected = json.loads(f.read())

    assert expected["name"] == output["name"]

    for key in (
        "psnr-y",
        "psnr-u",
        "psnr-v",
        "psnr-yuv",
        "psnr-rgb",
        "ms-ssim-rgb",
        "bitrate",
    ):
        if key not in expected["results"]:
            continue
        assert np.allclose(
            expected["results"][key], output["results"][key], rtol=1e-5, atol=1e-5
        )
