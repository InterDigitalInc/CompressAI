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

import argparse
import pickle
import re
import sys

# expect pickle file format:
# {
#   'quantizers': { 'weight0': 10, 'weight1': 13 ...},
# }
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="extract_quantizers",
        usage="use output of debug_model to identify layers index.\nusage: debug_model model.sadl | extract_quantizers --input model_info.pkl > quantizers.txt",
    )
    parser.add_argument(
        "--input",
        action="store",
        nargs="?",
        type=str,
        help="name of the pkl file containing the dict",
    )
    parser.add_argument("--verbose", action="count")

args = parser.parse_args()

with open(args.input, "rb") as f:
    infos = pickle.load(f)

if args.verbose:
    print(infos, file=sys.stderr)

started = False
d = {}
firstLayer = None
conv = {}
for line in sys.stdin:
    if "Exit" == line.rstrip() or "end model inference" in line:
        break
    if "start model inference" in line:
        started = True
    if started:
        m = re.match(r"\[INFO\]\s+(\d+)\s+Const\s+\((.+)\).*", line)
        if m is not None:
            d[m[2]] = int(m[1])

        m = re.match(r"\[INFO\]\s+(\d+)\s+Conv2DTranspose\s+\((.+)\).*", line)
        if m is not None:
            if firstLayer is None:
                firstLayer = int(m[1])
            else:
                conv[int(m[1])] = 0

if args.verbose:
    print(
        "[INFO] first layer: {}, layers index: {}".format(firstLayer, d),
        file=sys.stderr,
    )

assert "quantizers" in infos, "No quantizers key in {}".format(args.input)

# get quantizers
print("0 0 ", end="")  # input is already in integer, do not put any quantizer
for k, v in infos["quantizers"].items():
    name = k
    assert name in d, "{} not in extracted names {}".format(name, d)
    print("{} {} ".format(d[name], int(v)), end="")
    if int(d[name]) == firstLayer - 1:  # const layer corresponding to the first deconv
        print("{} {} ".format(firstLayer, -int(v)), end="")
for k, v in conv.items():
    print("{} {} ".format(k, v), end="")
print("")
