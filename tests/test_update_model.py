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

import importlib
import io

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
import torch

from compressai.models.google import FactorizedPrior

update_model_module = importlib.import_module("compressai.utils.update_model.__main__")


def run_update_model(*args):
    fout, ferr = io.StringIO(), io.StringIO()
    with redirect_stderr(ferr):
        with redirect_stdout(fout):
            update_model_module.main(map(str, args))
    return fout.getvalue(), ferr.getvalue()


def test_missing_filepath():
    with pytest.raises(SystemExit):
        run_update_model()


def test_invalid_filepath(tmpdir):
    # directory
    with pytest.raises(RuntimeError):
        run_update_model(tmpdir)

    # empty/invalid file
    p = tmpdir.join("hello.txt")
    p.write("")
    with pytest.raises((EOFError, TypeError)):
        run_update_model(p)


def test_valid(tmpdir):
    p = tmpdir.join("model.pth.tar").strpath

    net = FactorizedPrior(32, 64)
    torch.save(net.state_dict(), p)

    stdout, stderr = run_update_model(
        p, "--architecture", "factorized-prior", "--dir", tmpdir
    )
    assert len(stdout) == 0
    assert len(stderr) == 0

    files = list(Path(tmpdir).glob("*.pth.tar"))
    assert len(files) == 1

    cdf_len = net.state_dict()["entropy_bottleneck._cdf_length"]
    new_cdf_len = torch.load(files[0])["entropy_bottleneck._cdf_length"]
    assert cdf_len.size(0) != new_cdf_len.size(0)


def test_valid_name(tmpdir):
    p = tmpdir.join("model.pth.tar").strpath

    net = FactorizedPrior(32, 64)
    torch.save(net.state_dict(), p)

    stdout, stderr = run_update_model(
        p, "--architecture", "factorized-prior", "--dir", tmpdir, "--name", "yolo"
    )
    assert len(stdout) == 0
    assert len(stderr) == 0

    files = sorted(Path(tmpdir).glob("*.pth.tar"))
    assert len(files) == 2

    assert files[0].name == "model.pth.tar"
    assert files[1].name[:5] == "yolo-"


def test_valid_no_update(tmpdir):
    p = tmpdir.join("model.pth.tar").strpath

    net = FactorizedPrior(32, 64)
    torch.save(net.state_dict(), p)

    stdout, stderr = run_update_model(
        p, "--architecture", "factorized-prior", "--dir", tmpdir, "--no-update"
    )
    assert len(stdout) == 0
    assert len(stderr) == 0

    files = list(Path(tmpdir).glob("*.pth.tar"))
    assert len(files) == 1

    cdf_len = net.state_dict()["entropy_bottleneck._cdf_length"]
    new_cdf_len = torch.load(files[0])["entropy_bottleneck._cdf_length"]
    assert cdf_len.size(0) == new_cdf_len.size(0)


def test_invalid_model(tmpdir):
    p = tmpdir.join("model.pth.tar").strpath

    net = FactorizedPrior(32, 64)
    torch.save(net.state_dict(), p)

    with pytest.raises(SystemExit):
        run_update_model(p, "--architecture", "foobar")


def test_load(tmpdir):
    p = tmpdir.join("model.pth.tar").strpath

    net = FactorizedPrior(32, 64)

    for k in ["network", "state_dict"]:
        torch.save({k: net.state_dict()}, p)
        stdout, stderr = run_update_model(
            p, "--architecture", "factorized-prior", "--dir", tmpdir
        )
        assert len(stdout) == 0
        assert len(stderr) == 0
