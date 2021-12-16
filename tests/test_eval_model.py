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
update_model = importlib.import_module("compressai.utils.update_model.__main__")

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
    dirpath = os.path.join(here, "assets/dataset/image")

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
            expected["results"][key], output["results"][key], rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("model_name", ("factorized-prior", "bmshj2018-factorized"))
def test_eval_model_ckpt(tmp_path, model_name):
    here = os.path.dirname(__file__)
    parent = os.path.dirname(here)

    # fake training
    datapath = os.path.join(here, "assets/fakedata/imagefolder")
    spec = importlib.util.spec_from_file_location(
        "examples.train", os.path.join(parent, "examples/train.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        "-d",
        datapath,
        "-e",
        "1",
        "--batch-size",
        "1",
        "--patch-size",
        "48",
        "64",
        "--seed",
        "0",
        "--save",
    ]

    os.chdir(tmp_path)
    module.main(argv)

    checkpoint = "checkpoint_best_loss.pth.tar"
    assert os.path.isfile(checkpoint)

    # update model
    cmd = ["-a", model_name, "-n", "factorized", checkpoint]
    update_model.main(cmd)

    # ckpt evaluation
    dirpath = os.path.join(here, "assets/dataset/image")
    checkpoint = next(f for f in os.listdir(tmp_path) if f.startswith("factorized-"))
    cmd = [
        "checkpoint",
        dirpath,
        "-a",
        "bmshj2018-factorized",
        "-p",
        checkpoint,
    ]
    eval_model.main(cmd)
