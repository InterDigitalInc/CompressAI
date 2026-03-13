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

import importlib.util
import io
import os
import re
import subprocess
import sys

from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

# Example: GENERATE_EXPECTED=1 pytest -sx tests/test_eval_model.py
GENERATE_EXPECTED = os.getenv("GENERATE_EXPECTED")


def load_train_module(rootdir):
    spec = importlib.util.spec_from_file_location(
        "examples.train", rootdir / "examples/train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_train_example():
    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent

    module = load_train_module(rootdir)

    argv = [
        "-d",
        str(rootdir / "tests/assets/fakedata/imagefolder"),
        "-e",
        "10",
        "--batch-size",
        "1",
        "--patch-size",
        "48",
        "128",
        "--seed",
        "42",
        "--num-workers",
        "2",
    ]

    f = io.StringIO()
    with redirect_stdout(f):
        module.main(argv)
    log = f.getvalue()

    logpath = cwd / "expected" / "train_log_42.txt"
    if not logpath.is_file():
        if not GENERATE_EXPECTED:
            raise RuntimeError(f"Missing expected file {logpath}")
        with logpath.open("w") as f:
            f.write(log)

    with logpath.open("r") as f:
        expected = f.read()

    test_values = [m[0] for m in re.findall(r"(?P<number>([0-9]*[.])?[0-9]+)", log)]
    expected_values = [
        m[0] for m in re.findall(r"(?P<number>([0-9]*[.])?[0-9]+)", expected)
    ]

    assert len(test_values) == len(expected_values)
    for a, b in zip(test_values, expected_values):
        try:
            assert int(a) == int(b)
        except ValueError:
            assert float(a) == pytest.approx(float(b), rel=1e-3)


@pytest.mark.slow
def test_train_example_ddp(tmp_path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")

    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{rootdir}{os.pathsep}{pythonpath}" if pythonpath else str(rootdir)
    )

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        str(rootdir / "examples/train.py"),
        "-d",
        str(rootdir / "tests/assets/fakedata/imagefolder"),
        "-e",
        "1",
        "--batch-size",
        "1",
        "--test-batch-size",
        "1",
        "--patch-size",
        "48",
        "128",
        "--seed",
        "42",
        "--num-workers",
        "0",
    ]
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Learning rate:") == 1
    assert result.stdout.count("Test epoch 0: Average losses:") == 1
    assert "Train epoch 0:" in result.stdout

    checkpoint_path = tmp_path / "checkpoint.pth.tar"
    best_checkpoint_path = tmp_path / "checkpoint_best_loss.pth.tar"
    assert checkpoint_path.is_file()
    assert best_checkpoint_path.is_file()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["epoch"] == 0
    assert not any(key.startswith("module.") for key in checkpoint["state_dict"])
