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
"""
Collect performance metrics of published traditional or end-to-end image
codecs.
"""
from typing import List

import argparse
import json
import os
import sys

import multiprocessing as mp

from collections import defaultdict
from itertools import starmap

from .codecs import Codec, JPEG, WebP, JPEG2000, BPG, TFCI, VTM, HM, AV1

# from torchvision.datasets.folder
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')

codecs = [JPEG, WebP, JPEG2000, BPG, TFCI, VTM, HM, AV1]


# we need the quality index (not value) to compute the stats later
def func(codec, i, *args):
    rv = codec.run(*args)
    return i, rv


def collect(codec: Codec,
            dataset: str,
            qualities: List[int],
            num_jobs: int = 1):
    filepaths = [
        os.path.join(dataset, f) for f in os.listdir(dataset)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

    pool = mp.Pool(num_jobs) if num_jobs > 1 else None

    if len(filepaths) == 0:
        print('No images found in the dataset directory')
        sys.exit(1)

    args = [(codec, i, f, q) for i, q in enumerate(qualities)
            for f in filepaths]

    if pool:
        rv = pool.starmap(func, args)
    else:
        rv = list(starmap(func, args))

    results = [defaultdict(float) for _ in range(len(qualities))]
    for i, metrics in rv:
        for k, v in metrics.items():
            results[i][k] += v
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
    description = 'Collect codec metrics.'
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest='codec', help='Select codec')
    subparsers.required = True
    return parser, subparsers


def setup_common_args(parser):
    parser.add_argument('dataset', type=str)
    parser.add_argument('-j',
                        '--num-jobs',
                        type=int,
                        metavar='N',
                        default=1,
                        help='Number of parallel jobs (default: %(default)s)')
    parser.add_argument('-q',
                        '--quality',
                        dest='qualities',
                        metavar='',
                        default=[75],
                        nargs='*',
                        type=int,
                        help='quality parameter (default: %(default)s)')


def main(argv):
    parser, subparsers = setup_args()
    for c in codecs:
        cparser = subparsers.add_parser(c.__name__.lower(),
                                        help=f'{c.__name__}')
        setup_common_args(cparser)
        c.setup_args(cparser)
    args = parser.parse_args(argv)

    codec_cls = next(c for c in codecs if c.__name__.lower() == args.codec)
    codec = codec_cls(args)
    results = collect(codec, args.dataset, args.qualities, args.num_jobs)

    output = {
        'name': codec.name,
        'description': codec.description,
        'results': results,
    }

    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
