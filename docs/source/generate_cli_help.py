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

        doc = m.__doc__
        if doc:
            fout.write(doc)
            fout.write('\n')

        fout.write('.. code-block:: text\n\n')
        capture = io.StringIO()
        parser = m.setup_args()
        if isinstance(parser, tuple):
            parser = parser[0]
        parser.prog = f'python -m compression.utils.{p.name}'
        parser.print_help(capture)

        for line in capture.getvalue().split('\n'):
            fout.write(f'\t{line}\n')

        fout.write('\n\n')

    fout.close()


if __name__ == '__main__':
    main()
