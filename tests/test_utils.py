import sys

from pathlib import Path

import subprocess
from subprocess import Popen

import torch

from compressai.models.priors import FactorizedPrior


class TestUpdate:
    def _run_update(self, *args, timeout=16):
        cmd = [sys.executable, '-m', 'compressai.utils.update_model']
        cmd += list(args)
        p = Popen(cmd,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE,
                  cwd=Path(__file__).parent)
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        rc = p.returncode
        out = out.decode('ascii')
        err = err.decode('ascii')
        return (rc, out, err)

    def test_missing_filepath(self):
        rc, _, _ = self._run_update()
        assert rc == 2

    def test_invalid_filepath(self, tmpdir):
        rc, _, _ = self._run_update(tmpdir)
        assert rc == 1

        p = tmpdir.join('hello.txt')
        p.write('')
        rc, _, _ = self._run_update(p)
        assert rc == 1

    def test_valid(self, tmpdir):
        p = tmpdir.join('model.pth.tar').strpath

        net = FactorizedPrior(32, 64)
        torch.save(net.state_dict(), p)

        rc, _, _ = self._run_update(p, '--architecture', 'factorized-prior',
                                    '--dir', tmpdir)
        assert rc == 0

        files = list(Path(tmpdir).glob('*.pth.tar'))
        assert len(files) == 1

        cdf_len = net.state_dict()['entropy_bottleneck._cdf_length']
        new_cdf_len = torch.load(files[0])['entropy_bottleneck._cdf_length']
        assert cdf_len.size(0) != new_cdf_len.size(0)

    def test_valid_no_update(self, tmpdir):
        p = tmpdir.join('model.pth.tar').strpath

        net = FactorizedPrior(32, 64)
        torch.save(net.state_dict(), p)

        rc, _, _ = self._run_update(p, '--architecture', 'factorized-prior',
                                    '--dir', tmpdir, '--no-update')
        assert rc == 0

        files = list(Path(tmpdir).glob('*.pth.tar'))
        assert len(files) == 1

        cdf_len = net.state_dict()['entropy_bottleneck._cdf_length']
        new_cdf_len = torch.load(files[0])['entropy_bottleneck._cdf_length']
        assert cdf_len.size(0) == new_cdf_len.size(0)

    def test_invalid_model(self, tmpdir):
        p = tmpdir.join('model.pth.tar').strpath

        net = FactorizedPrior(32, 64)
        torch.save(net.state_dict(), p)

        rc, _, _ = self._run_update(p, '--architecture', 'foobar')
        assert rc == 2

    def test_load(self, tmpdir):
        p = tmpdir.join('model.pth.tar').strpath

        net = FactorizedPrior(32, 64)

        for k in ['network', 'state_dict']:
            torch.save({k: net.state_dict()}, p)
            rc, _, _ = self._run_update(p, '--architecture',
                                        'factorized-prior', '--dir', tmpdir)
            assert rc == 0
