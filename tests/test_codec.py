# Copyright (c) 2021-2024, InterDigital Communications, Inc
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

import hashlib
import importlib.util
import itertools
import os

from pathlib import Path

import pytest
import torch

from compressai.zoo import image_models

# Example: GENERATE_EXPECTED=1 pytest -sx tests/test_bench_codec_video.py
GENERATE_EXPECTED = os.getenv("GENERATE_EXPECTED")

archs = [
    "bmshj2018-factorized",
    "bmshj2018-hyperprior",
    "mbt2018-mean",
    "mbt2018",
]


class TestCompressDecompress:
    @pytest.mark.parametrize("arch,N", itertools.product(archs, [1, 2, 3]))
    def test_image_codec(self, arch: str, N: int):
        x = torch.zeros(N, 3, 256, 256)
        h, w = x.size()[-2:]
        x[:, :, h // 4 : -h // 4, w // 4 : -w // 4].fill_(1)

        model = image_models[arch]
        net = model(quality=1, metric="mse", pretrained=True, progress=False).eval()
        with torch.no_grad():
            rv = net.compress(x)

        assert "shape" in rv
        shape = rv["shape"]
        ds = net.downsampling_factor
        assert shape == torch.Size([x.size(2) // ds, x.size(3) // ds])

        assert "strings" in rv
        strings_list = rv["strings"]
        # y_strings (+ optional z_strings)
        assert len(strings_list) == 1 or len(strings_list) == 2
        for strings in strings_list:
            assert len(strings) == N
            for string in strings:
                assert isinstance(string, bytes)
                assert len(string) > 0

        with torch.no_grad():
            rv = net.decompress(strings_list, shape)
        assert "x_hat" in rv
        x_hat = rv["x_hat"]
        assert x_hat.size() == x.size()

        mse = torch.mean((x - x_hat) ** 2)
        psnr = -10 * torch.log10(mse).item()
        assert 35 < psnr < 41


class TestCodecExample:
    @pytest.mark.skip(reason="find a better way to test this")
    @pytest.mark.parametrize("model", ("bmshj2018-factorized",))
    def test_encode_decode_image(self, tmpdir, model):
        cwd = Path(__file__).resolve().parent
        rootdir = cwd.parent

        spec = importlib.util.spec_from_file_location(
            "examples.codec", rootdir / "examples/codec.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inputpath = str(rootdir / "tests/assets/dataset/image/stmalo_fracape.png")
        binpath = f"{tmpdir}/{model}_stmalo_fracape.bin"
        argv = [
            "encode",
            inputpath,
            "--model",
            model,
            "-o",
            binpath,
        ]
        module.main(argv)

        md5sum_bin = hashlib.md5(open(binpath, "rb").read()).hexdigest()
        expected_md5sum_bin_file = (
            cwd / "expected" / f"md5sum-bin-{model}-stmalo_fracape.txt"
        )
        if not expected_md5sum_bin_file.is_file():
            if not GENERATE_EXPECTED:
                raise RuntimeError(f"Missing expected file {expected_md5sum_bin_file}")

            with expected_md5sum_bin_file.open("wt") as f:
                f.write(md5sum_bin)

        with expected_md5sum_bin_file.open("r") as f:
            expected_md5sum_bin = f.read()

        assert expected_md5sum_bin == md5sum_bin

        decpath = f"{tmpdir}/{model}_dec_stmalo_fracape.png"
        argv = [
            "decode",
            binpath,
            "-o",
            decpath,
        ]
        module.main(argv)

        md5sum_dec = hashlib.md5(open(decpath, "rb").read()).hexdigest()
        expected_md5sum_dec_file = (
            cwd / "expected" / f"md5sum-dec-model-{model}-stmalo_fracape.txt"
        )
        if not expected_md5sum_dec_file.is_file():
            if not GENERATE_EXPECTED:
                raise RuntimeError(f"Missing expected file {expected_md5sum_dec_file}")

            with expected_md5sum_dec_file.open("wt") as f:
                f.write(md5sum_dec)

        with expected_md5sum_dec_file.open("r") as f:
            expected_md5sum_dec = f.read()

        assert expected_md5sum_dec == md5sum_dec

    @pytest.mark.skip(reason="find a better way to test this")
    @pytest.mark.parametrize("model", ("ssf2020",))
    @pytest.mark.parametrize("nb_frames", ("1",))
    def test_encode_decode_video(self, tmpdir, model, nb_frames):
        cwd = Path(__file__).resolve().parent
        rootdir = cwd.parent

        spec = importlib.util.spec_from_file_location(
            "examples.codec", rootdir / "examples/codec.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inputpath = str(
            rootdir
            / "tests/assets/dataset/video/C_RaceHorses_2frames_832x480_30Hz_8bit_P420.yuv"
        )
        binpath = f"{tmpdir}/{model}_RaceHorses_{nb_frames}fr.bin"
        argv = [
            "encode",
            inputpath,
            "--model",
            model,
            "-o",
            binpath,
            "-f",
            nb_frames,
        ]
        module.main(argv)

        md5sum_bin = hashlib.md5(open(binpath, "rb").read()).hexdigest()
        expected_md5sum_bin_file = (
            cwd / "expected" / f"md5sum-bin-{model}-RaceHorses-{nb_frames}fr.txt"
        )
        if not expected_md5sum_bin_file.is_file():
            if not GENERATE_EXPECTED:
                raise RuntimeError(f"Missing expected file {expected_md5sum_bin_file}")

            with expected_md5sum_bin_file.open("wt") as f:
                f.write(md5sum_bin)

        with expected_md5sum_bin_file.open("r") as f:
            expected_md5sum_bin = f.read()

        assert expected_md5sum_bin == md5sum_bin

        decpath = f"{tmpdir}/{model}_dec_C_RaceHorses_{nb_frames}fr.yuv"
        argv = [
            "decode",
            binpath,
            "-o",
            decpath,
        ]
        module.main(argv)

        md5sum_dec = hashlib.md5(open(decpath, "rb").read()).hexdigest()
        expected_md5sum_dec_file = (
            cwd / "expected" / f"md5sum-dec-model-{model}-RaceHorses_{nb_frames}fr.txt"
        )
        if not expected_md5sum_dec_file.is_file():
            if not GENERATE_EXPECTED:
                raise RuntimeError(f"Missing expected file {expected_md5sum_dec_file}")

            with expected_md5sum_dec_file.open("wt") as f:
                f.write(md5sum_dec)

        with expected_md5sum_dec_file.open("r") as f:
            expected_md5sum_dec = f.read()

        assert expected_md5sum_dec == md5sum_dec
