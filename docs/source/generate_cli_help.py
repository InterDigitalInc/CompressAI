# Based on https://github.com/facebookresearch/ParlAI/tree/c06c40603f45918f58cb09122fa8c74dd4047057/docs/source

import importlib
import io

from pathlib import Path

import compressai.utils


def get_utils():
    rootdir = Path(compressai.utils.__file__).parent
    for d in rootdir.iterdir():
        if d.is_dir() and (d / '__main__.py').is_file():
            yield d


def main():
    fout = open('cli_usage.inc', 'w')

    for p in get_utils():
        try:
            m = importlib.import_module(f'compressai.utils.{p.name}.__main__')
        except ImportError:
            continue

        if not hasattr(m, 'setup_args'):
            continue

        fout.write(p.name)
        fout.write('\n')
        fout.write('-' * len(p.name))
        fout.write('\n')

        fout.write('.. code-block:: text\n\n')
        capture = io.StringIO()
        parser = m.setup_args()
        parser.prog = f'python -m compression.utils.{p.name}'
        parser.print_help(capture)

        for line in capture.getvalue().split('\n'):
            fout.write(f'\t{line}\n')

        fout.write('\n\n')

    fout.close()


if __name__ == '__main__':
    main()
