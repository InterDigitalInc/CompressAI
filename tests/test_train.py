import importlib.util
import io

from contextlib import redirect_stdout
from pathlib import Path


def test_train_example():
    cwd = Path(__file__).resolve().parent
    rootdir = cwd.parent

    spec = importlib.util.spec_from_file_location(
        'examples.train', rootdir / 'examples/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        '-d',
        str(rootdir / 'tests/assets/fakedata/imagefolder'),
        '-e',
        '10',
        '--batch-size',
        '1',
        '--patch-size',
        '48',
        '128',
        '--seed',
        '3.14',
    ]

    f = io.StringIO()
    with redirect_stdout(f):
        module.main(argv)
    log = f.getvalue()

    logpath = cwd / 'expected' / 'train_log_3.14.txt'
    with logpath.open('r') as f:
        expected = f.read()

    assert log == expected
