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

# Based on https://github.com/facebookresearch/ParlAI/tree/c06c40603f45918f58cb09122fa8c74dd4047057/docs/source

import importlib
import io

from pathlib import Path

import compressai.utils


def get_utils():
    rootdir = Path(compressai.utils.__file__).parent
    for d in sorted(rootdir.iterdir()):
        if d.is_dir() and (d / "__main__.py").is_file():
            yield d


def main():
    fout = open("cli_usage.inc", "w")

    for p in get_utils():
        try:
            m = importlib.import_module(f"compressai.utils.{p.name}.__main__")
        except ImportError:
            continue

        if not hasattr(m, "setup_args"):
            continue

        fout.write(p.name)
        fout.write("\n")
        fout.write("-" * len(p.name))
        fout.write("\n")

        doc = m.__doc__
        if doc:
            fout.write(doc)
            fout.write("\n")

        fout.write(".. code-block:: text\n\n")
        capture = io.StringIO()
        parser = m.setup_args()
        if isinstance(parser, tuple):
            parser = parser[0]
        parser.prog = f"python -m compressai.utils.{p.name}"
        parser.print_help(capture)

        for line in capture.getvalue().split("\n"):
            fout.write(f"\t{line}\n")

        fout.write("\n\n")

    fout.close()


if __name__ == "__main__":
    main()
