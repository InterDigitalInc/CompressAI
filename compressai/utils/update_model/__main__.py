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
Update the CDFs parameters of a trained model.

To be called on a model checkpoint after training. This will update the internal
CDFs related buffers required for entropy coding.
"""
from typing import Dict

import argparse
import hashlib
import sys

from pathlib import Path

import torch

from compressai.models.priors import (FactorizedPrior, MeanScaleHyperprior,
                                      ScaleHyperprior,
                                      JointAutoregressiveHierarchicalPriors)


def sha256_file(filepath: Path, len_hash_prefix: int = 8) -> str:
    # from pytorch github repo
    sha256 = hashlib.sha256()
    with filepath.open('rb') as f:
        while True:
            buf = f.read(8192)
            if len(buf) == 0:
                break
            sha256.update(buf)
    digest = sha256.hexdigest()

    return digest[:len_hash_prefix]


def load_checkpoint(filepath: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(filepath, map_location='cpu')

    if 'network' in checkpoint:
        state_dict = checkpoint['network']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    return state_dict


description = """
Export a trained model to a new checkpoint with an updated CDFs parameters and a 
hash prefix, so that it can be loaded later via `load_state_dict_from_url`.
""".strip()

models = {
    'factorized-prior': FactorizedPrior,
    'jarhp': JointAutoregressiveHierarchicalPriors,
    'mean-scale-hyperprior': MeanScaleHyperprior,
    'scale-hyperprior': ScaleHyperprior,
}


def setup_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('filepath',
                        type=str,
                        help='Path to the checkpoint model to be exported.')
    parser.add_argument('-n', '--name', type=str, help='Exported model name.')
    parser.add_argument('-d',
                        '--dir',
                        type=str,
                        help='Exported model directory.')
    parser.add_argument('--no-update',
                        action='store_true',
                        default=False,
                        help='Do not update the model CDFs parameters.')
    parser.add_argument('--architecture',
                        default='scale-hyperprior',
                        choices=models.keys(),
                        help='Set model architecture (default: %(default)s).')
    return parser


def main(argv):
    args = setup_args().parse_args(argv)

    filepath = Path(args.filepath).resolve()
    if not filepath.is_file():
        raise RuntimeError(f'"{filepath}" is not a valid file.')

    state_dict = load_checkpoint(filepath)

    model_cls = models[args.architecture]
    net = model_cls.from_state_dict(state_dict)

    if not args.no_update:
        net.update(force=True)
    state_dict = net.state_dict()

    if not args.name:
        filename = filepath
        while filename.suffixes:
            filename = Path(filename.stem)
    else:
        filename = args.name

    ext = ''.join(filepath.suffixes)

    if args.dir is not None:
        output_dir = args.dir
        Path(output_dir).mkdir(exist_ok=True)
    else:
        output_dir = Path.cwd().name

    filepath = Path(f'{output_dir}/{filename}{ext}')
    torch.save(state_dict, filepath)
    hash_prefix = sha256_file(filepath)

    filepath.rename(f'{output_dir}/{filename}-{hash_prefix}{ext}')


if __name__ == '__main__':
    main(sys.argv[1:])
