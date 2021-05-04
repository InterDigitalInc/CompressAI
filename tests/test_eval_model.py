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
import random

import numpy as np
import pytest
import torch

eval_model = importlib.import_module("compressai.utils.eval_model.__main__")

# Example: GENERATE_EXPECTED=1 pytest -sx tests/test_eval_model.py
GENERATE_EXPECTED = os.getenv("GENERATE_EXPECTED")


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def test_eval_model():
    with pytest.raises(SystemExit):
        eval_model.main(["--help"])

    with pytest.raises(SystemExit):
        eval_model.main([])

    with pytest.raises(SystemExit):
        eval_model.main(["pretrained"])

    with pytest.raises(SystemExit):
        eval_model.main(
            [
                "pretrained",
                ".",
                "-a",
                "bmshj2018-factorized",
                "-m",
                "mse",
                "-q",
                "1",
            ]
        )


@pytest.mark.parametrize("model", ("bmshj2018-factorized",))
@pytest.mark.parametrize("quality", ("1", "4", "8"))
@pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
@pytest.mark.parametrize("entropy_estimation", (False, True))
def test_eval_model_pretrained(capsys, model, quality, metric, entropy_estimation):
    here = os.path.dirname(__file__)
    dirpath = os.path.join(here, "assets/dataset")

    cmd = [
        "pretrained",
        dirpath,
        "-a",
        model,
        "-m",
        metric,
        "-q",
        quality,
    ]
    if entropy_estimation:
        cmd += ["--entropy-estimation"]
    eval_model.main(cmd)

    output = capsys.readouterr().out
    output = json.loads(output)
    expected = os.path.join(
        here,
        "expected",
        f"eval_{int(entropy_estimation)}_{model}_{metric}_{quality}.json",
    )

    if not os.path.isfile(expected):
        if not GENERATE_EXPECTED:
            raise RuntimeError(f"Missing expected file {expected}")
        with open(expected, "w") as f:
            json.dump(output, f)

    with open(expected, "r") as f:
        expected = json.loads(f.read())

    for key in ("name", "description"):
        assert expected[key] == output[key]

    for key in ("psnr", "ms-ssim", "bpp"):
        if key not in expected["results"]:
            continue
        assert np.allclose(
            expected["results"][key], output["results"][key], rtol=1e-5, atol=1e-5
        )
