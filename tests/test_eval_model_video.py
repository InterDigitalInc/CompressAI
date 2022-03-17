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
import random

import numpy as np
import pytest
import torch

eval_model = importlib.import_module("compressai.utils.video.eval_model.__main__")
update_model = importlib.import_module("compressai.utils.update_model.__main__")

# Example: GENERATE_EXPECTED=1 pytest -sx tests/test_eval_model.py
GENERATE_EXPECTED = os.getenv("GENERATE_EXPECTED")


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def test_eval_model_video():
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
                "ssf2020",
                "-m",
                "mse",
                "-q",
                "1",
            ]
        )


# mse and entropy_estimation tested for now
@pytest.mark.parametrize("model", ("ssf2020",))
@pytest.mark.parametrize("quality", ("1", "4", "8"))
@pytest.mark.parametrize("metric", ("mse",))
@pytest.mark.parametrize("entropy_estimation", (True, False))
def test_eval_model_pretrained(
    capsys, model, quality, metric, entropy_estimation, tmpdir
):
    here = os.path.dirname(__file__)
    dirpath = os.path.join(here, "assets/dataset/video")

    cmd = [
        "pretrained",
        dirpath,
        str(tmpdir),
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
            expected["results"][key], output["results"][key], rtol=1e-4, atol=1e-4
        )


# @pytest.mark.parametrize("model_name", ("ssf2020",))
# def test_eval_model_ckpt(tmp_path, model_name):
#     here = os.path.dirname(__file__)
#     parent = os.path.dirname(here)

#     # fake training
#     datapath = os.path.join(here, "assets/fakedata/imagefolder")
#     spec = importlib.util.spec_from_file_location(
#         "examples.train", os.path.join(parent, "examples/train_video.py")
#     )
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)

#     argv = [
#         "-d",
#         datapath,
#         "-e",
#         "1",
#         "--batch-size",
#         "1",
#         "--patch-size",
#         "48",
#         "64",
#         "--seed",
#         "0",
#         "--save",
#     ]

#     os.chdir(tmp_path)
#     module.main(argv)

#     checkpoint = "checkpoint_best_loss.pth.tar"
#     assert os.path.isfile(checkpoint)

#     # update model
#     cmd = ["-a", model_name, "-n", "factorized", checkpoint]
#     update_model.main(cmd)

#     # ckpt evaluation
#     dirpath = os.path.join(here, "assets/dataset/image")
#     checkpoint = next(f for f in os.listdir(tmp_path) if f.startswith("factorized-"))
#     cmd = [
#         "checkpoint",
#         dirpath,
#         "-a",
#         "bmshj2018-factorized",
#         "-p",
#         checkpoint,
#     ]
#     eval_model.main(cmd)
