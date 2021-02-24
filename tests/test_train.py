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

import importlib.util
import io
import re

from contextlib import redirect_stdout
from pathlib import Path

import pytest


@pytest.mark.slow
def test_train_example():
    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent

    spec = importlib.util.spec_from_file_location(
        "examples.train", rootdir / "examples/train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
        "3.14",
    ]

    f = io.StringIO()
    with redirect_stdout(f):
        module.main(argv)
    log = f.getvalue()

    logpath = cwd / "expected" / "train_log_3.14.txt"
    if not logpath.is_file():
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
            assert pytest.approx(float(a), float(b))
