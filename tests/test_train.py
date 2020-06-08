import subprocess
import sys

from pathlib import Path


def test_train_example():
    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent

    rv = subprocess.check_output([
        sys.executable,
        rootdir / 'examples/train.py',
        '-d',
        rootdir / 'tests/assets/fakedata/imagefolder',
        '-e',
        '10',
        '--batch-size',
        '1',
        '--patch-size',
        '48',
        '128',
        '--seed',
        '3.14',
    ])

    logpath = cwd / 'expected' / 'train_log_3.14.txt'
    with logpath.open('r') as f:
        reflog = f.read()

    assert rv.decode('utf-8') == reflog
