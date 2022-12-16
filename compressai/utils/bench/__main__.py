# Copyright (c) 2021-2022, InterDigital Communications, Inc
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
Collect performance metrics of published traditional or end-to-end image
codecs.
"""
import argparse
import json
import multiprocessing as mp
import sys

from collections import defaultdict
from itertools import starmap
from pathlib import Path
from typing import List

from .codecs import AV1, BPG, HM, JPEG, JPEG2000, TFCI, VTM, Codec, WebP

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

codecs = [JPEG, WebP, JPEG2000, BPG, TFCI, VTM, HM, AV1]


# we need the quality index (not value) to compute the stats later
def func(codec, i, *args):
    rv = codec.run(*args)
    return i, rv


def collect(
    codec: Codec,
    dataset: str,
    qps: List[int],
    metrics: List[str],
    num_jobs: int = 1,
):
    if not Path(dataset).is_dir():
        raise OSError(f"No such directory: {dataset}")

    filepaths = []
    for ext in IMG_EXTENSIONS:
        filepaths.extend(Path(dataset).rglob(f"*{ext}"))

    pool = mp.Pool(num_jobs) if num_jobs > 1 else None

    if len(filepaths) == 0:
        print("No images found in the dataset directory")
        sys.exit(1)

    args = [
        (codec, i, f, q, metrics) for i, q in enumerate(qps) for f in sorted(filepaths)
    ]

    if pool:
        rv = pool.starmap(func, args)
    else:
        rv = list(starmap(func, args))

    results = [defaultdict(float) for _ in range(len(qps))]

    for i, metrics in rv:
        for k, v in metrics.items():
            results[i][k] += v

    # aggregate results for all images
    for i, _ in enumerate(results):
        for k, v in results[i].items():
            results[i][k] = v / len(filepaths)

    # list of dict -> dict of list
    out = defaultdict(list)
    for r in results:
        for k, v in r.items():
            out[k].append(v)
    return out


def setup_args():
    description = "Collect codec metrics."
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="codec", help="Select codec")
    subparsers.required = True
    return parser, subparsers


def setup_common_args(parser):
    parser.add_argument("dataset", type=str)
    parser.add_argument(
        "-j",
        "--num-jobs",
        type=int,
        metavar="N",
        default=1,
        help="number of parallel jobs (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--qps",
        dest="qps",
        type=str,
        default="75",
        help="list of quality/quantization parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        dest="metrics",
        default=["psnr", "ms-ssim"],
        nargs="+",
        help="do not return PSNR and MS-SSIM metrics (use for very small images)",
    )


def main(argv):
    parser, subparsers = setup_args()
    for c in codecs:
        cparser = subparsers.add_parser(c.__name__.lower(), help=f"{c.__name__}")
        setup_common_args(cparser)
        c.setup_args(cparser)
    args = parser.parse_args(argv)

    codec_cls = next(c for c in codecs if c.__name__.lower() == args.codec)
    codec = codec_cls(args)
    qps = [int(q) for q in args.qps.split(",") if q]
    results = collect(
        codec,
        args.dataset,
        sorted(qps),
        args.metrics,
        args.num_jobs,
    )

    output = {
        "name": codec.name,
        "description": codec.description,
        "results": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
