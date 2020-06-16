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
import io

from contextlib import redirect_stdout, redirect_stderr

from pathlib import Path

import torch

import pytest

from compressai.models.priors import FactorizedPrior

update_model_module = importlib.import_module(
    'compressai.utils.update_model.__main__')


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
    p = tmpdir.join('hello.txt')
    p.write('')
    with pytest.raises((EOFError, TypeError)):
        run_update_model(p)


def test_valid(tmpdir):
    p = tmpdir.join('model.pth.tar').strpath

    net = FactorizedPrior(32, 64)
    torch.save(net.state_dict(), p)

    stdout, stderr = run_update_model(p, '--architecture', 'factorized-prior',
                                      '--dir', tmpdir)
    assert len(stdout) == 0
    assert len(stderr) == 0

    files = list(Path(tmpdir).glob('*.pth.tar'))
    assert len(files) == 1

    cdf_len = net.state_dict()['entropy_bottleneck._cdf_length']
    new_cdf_len = torch.load(files[0])['entropy_bottleneck._cdf_length']
    assert cdf_len.size(0) != new_cdf_len.size(0)


def test_valid_name(tmpdir):
    p = tmpdir.join('model.pth.tar').strpath

    net = FactorizedPrior(32, 64)
    torch.save(net.state_dict(), p)

    stdout, stderr = run_update_model(p, '--architecture', 'factorized-prior',
                                      '--dir', tmpdir, '--name', 'yolo')
    assert len(stdout) == 0
    assert len(stderr) == 0

    files = sorted(list(Path(tmpdir).glob('*.pth.tar')))
    assert len(files) == 2

    assert files[0].name == 'model.pth.tar'
    assert files[1].name[:5] == 'yolo-'


def test_valid_no_update(tmpdir):
    p = tmpdir.join('model.pth.tar').strpath

    net = FactorizedPrior(32, 64)
    torch.save(net.state_dict(), p)

    stdout, stderr = run_update_model(p, '--architecture', 'factorized-prior',
                                      '--dir', tmpdir, '--no-update')
    assert len(stdout) == 0
    assert len(stderr) == 0

    files = list(Path(tmpdir).glob('*.pth.tar'))
    assert len(files) == 1

    cdf_len = net.state_dict()['entropy_bottleneck._cdf_length']
    new_cdf_len = torch.load(files[0])['entropy_bottleneck._cdf_length']
    assert cdf_len.size(0) == new_cdf_len.size(0)


def test_invalid_model(tmpdir):
    p = tmpdir.join('model.pth.tar').strpath

    net = FactorizedPrior(32, 64)
    torch.save(net.state_dict(), p)

    with pytest.raises(SystemExit):
        run_update_model(p, '--architecture', 'foobar')


def test_load(tmpdir):
    p = tmpdir.join('model.pth.tar').strpath

    net = FactorizedPrior(32, 64)

    for k in ['network', 'state_dict']:
        torch.save({k: net.state_dict()}, p)
        stdout, stderr = run_update_model(p, '--architecture',
                                          'factorized-prior', '--dir', tmpdir)
        assert len(stdout) == 0
        assert len(stderr) == 0
