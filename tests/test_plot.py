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
import os

import matplotlib
import pytest

matplotlib.use("Agg")

plot = importlib.import_module("compressai.utils.plot.__main__")


@pytest.mark.parametrize("metric", ("psnr", "ms-ssim"))
def test_plot(metric):
    here = os.path.dirname(__file__)
    filepath = os.path.join(here, "expected/eval_0_bmshj2018-factorized_mse_1.json")
    cmd = ["-f", filepath, "--title", "myplot", "--metric", metric]
    plot.main(cmd)
