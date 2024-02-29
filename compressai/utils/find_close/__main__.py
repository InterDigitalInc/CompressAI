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

"""
Find the closest codec quality parameter to reach a given metric (bpp, ms-ssim,
or psnr).

Example usages:
    * :code:`python -m compressai.utils.find_close webp ~/picture.png 0.5 --metric bpp`
    * :code:`python -m compressai.utils.find_close jpeg ~/picture.png 35 --metric psnr --save`
"""

import argparse
import sys

from typing import Dict, List, Tuple

from PIL import Image

from compressai.utils.bench.codecs import AV1, BPG, HM, JPEG, JPEG2000, VTM, Codec, WebP


def get_codec_q_bounds(codec: Codec) -> Tuple[bool, int, int]:
    rev = False  # higher Q -> better quality or reverse
    if isinstance(codec, (JPEG, JPEG2000, WebP)):
        lower = -1
        upper = 101
    elif isinstance(codec, (BPG, HM)):
        lower = -1
        upper = 51
        rev = True
    elif isinstance(codec, (AV1, VTM)):
        lower = -1
        upper = 64
        rev = True
    else:
        raise ValueError(f"Invalid codec {codec}")
    return rev, lower, upper


def find_closest(
    codec: Codec, img: str, target: float, metric: str = "psnr-rgb"
) -> Tuple[int, Dict[str, float], Image.Image]:
    rev, lower, upper = get_codec_q_bounds(codec)

    best_rv = {}
    best_rec = None
    while upper > lower + 1:
        mid = (upper + lower) // 2
        rv, rec = codec.run(img, mid, return_rec=True)
        is_best = best_rv == {} or abs(rv[metric] - target) < abs(
            best_rv[metric] - target
        )
        if is_best:
            best_rv = rv
            best_rec = rec
        if rv[metric] > target:
            if not rev:
                upper = mid
            else:
                lower = mid
        elif rv[metric] < target:
            if not rev:
                lower = mid
            else:
                upper = mid
        else:
            break
        sys.stderr.write(
            f"\rtarget {metric}: {target:.4f} | value: {rv[metric]:.4f} | q: {mid}"
        )
        sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()
    return mid, best_rv, best_rec


codecs = [JPEG, WebP, JPEG2000, BPG, VTM, HM, AV1]


def setup_args():
    description = "Collect codec metrics and performances."
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="codec", help="Select codec")
    subparsers.required = True
    parser.add_argument("image", type=str, help="image filepath")
    parser.add_argument("target", type=float, help="target value to match")
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["bpp", "psnr-rgb", "ms-ssim-rgb"],
        default="bpp",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save reconstructed image to disk"
    )
    return parser, subparsers


def main(argv: List[str]):
    parser, subparsers = setup_args()
    for c in codecs:
        cparser = subparsers.add_parser(c.__name__.lower(), help=f"{c.__name__}")
        c.setup_args(cparser)
    args = parser.parse_args(argv)

    codec_cls = next(c for c in codecs if c.__name__.lower() == args.codec)
    codec = codec_cls(args)

    quality, metrics, rec = find_closest(codec, args.image, args.target, args.metric)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if args.save:
        rec.save(f"output_{codec_cls.__name__.lower()}_q{quality}.png")


if __name__ == "__main__":
    main(sys.argv[1:])
