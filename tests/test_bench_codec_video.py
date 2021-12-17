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
import json
import os

from pathlib import Path

import numpy as np
import pytest

bench_codec = importlib.import_module("compressai.utils.video.bench.__main__")

# Example: GENERATE_EXPECTED=1 pytest -sx tests/test_bench_codec_video.py
GENERATE_EXPECTED = os.getenv("GENERATE_EXPECTED")


def test_eval_model_cmd():
    with pytest.raises(SystemExit):
        bench_codec.main([])

    with pytest.raises(SystemExit):
        bench_codec.main(["x264"])


@pytest.mark.parametrize("codec", ("x264",))
def test_bench_codec_video(capsys, codec):
    here = Path(__file__).parent
    input_dir_path = here / "assets/dataset/video"
    output_dir_path = here / "assets/dataset/video"

    cmd = [codec, input_dir_path.as_posix(), output_dir_path.as_posix()]

    bench_codec.main(cmd)

    output = capsys.readouterr().out
    print(output)
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

    for key in ("psnr", "ms-ssim", "bpp"):
        if key not in expected["results"]:
            continue
        assert np.allclose(
            expected["results"][key], output["results"][key], rtol=1e-5, atol=1e-5
        )
