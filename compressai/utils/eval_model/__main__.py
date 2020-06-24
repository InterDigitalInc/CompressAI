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
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time

from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

from pytorch_msssim import ms_ssim

import compressai

from compressai.zoo import models

# from torchvision.datasets.folder
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert('RGB')
    return transforms.ToTensor()(img)


def inference(model, x):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x, (padding_left, padding_right, padding_top, padding_bottom),
        mode='constant',
        value=0)

    with torch.no_grad():
        start = time.time()
        out_enc = model.compress(x_padded)
        enc_time = time.time() - start

        start = time.time()
        out_dec = model.decompress(out_enc['strings'], out_enc['shape'])
        dec_time = time.time() - start

    out_dec['x_hat'] = F.pad(
        out_dec['x_hat'],
        (-padding_left, -padding_right, -padding_top, -padding_bottom))

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc['strings']) * 8. / num_pixels

    return {
        'psnr': psnr(x, out_dec['x_hat']),
        'msssim': ms_ssim(x, out_dec['x_hat'], data_range=1.).item(),
        'bpp': bpp,
        'encoding_time': enc_time,
        'decoding_time': dec_time,
    }


def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    with torch.no_grad():
        start = time.time()
        out_net = model.forward(x)
        elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
              for likelihoods in out_net['likelihoods'].values())

    return {
        'psnr': psnr(x, out_net['x_hat']),
        'bpp': bpp.item(),
        'encoding_time': elapsed_time / 2.,  # broad estimation
        'decoding_time': elapsed_time / 2.,
    }


def run_model(model, dataset, entropy_estimation=False):
    filepaths = [
        os.path.join(dataset, f) for f in os.listdir(dataset)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

    if len(filepaths) == 0:
        print('No images found in the dataset directory')
        sys.exit(1)

    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for f in filepaths:
        x = read_image(f).to(device)
        if not entropy_estimation:
            rv = inference(model, x)
        else:
            rv = inference_entropy_estimation(model, x)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():
    parser = argparse.ArgumentParser(description='Run model on image dataset')
    parser.add_argument('model',
                        type=str,
                        choices=models.keys(),
                        help='model architecture')
    parser.add_argument('-m',
                        '--metric',
                        type=str,
                        choices=['mse'],
                        default='mse',
                        help='metric trained against (default: %(default)s)')
    parser.add_argument('dataset', type=str, help='dataset path')
    parser.add_argument('-c',
                        '--entropy-coder',
                        choices=compressai.available_entropy_coders(),
                        default=compressai.available_entropy_coders()[0],
                        help='Entropy coder (default: %(default)s)')
    parser.add_argument('-q',
                        '--quality',
                        dest='qualities',
                        nargs='+',
                        type=int,
                        default=range(1, 9))
    parser.add_argument(
        '--entropy-estimation',
        action='store_true',
        help='Use evaluated entropy estimation (no entropy coding)')
    return parser


def main(argv):
    args = setup_args().parse_args(argv)

    compressai.set_entropy_coder(args.entropy_coder)

    results = defaultdict(list)
    for q in args.qualities:
        sys.stderr.write(f'\r{args.model} | quality: {q:d}')
        sys.stderr.flush()
        model = models[args.model](quality=q,
                                   metric=args.metric,
                                   pretrained=True).eval()
        metrics = run_model(model, args.dataset, args.entropy_estimation)
        for k, v in metrics.items():
            results[k].append(v)
    sys.stderr.write('\n')
    sys.stderr.flush()

    description = 'entropy estimation' \
        if args.entropy_estimation else args.entropy_coder
    output = {
        'name': args.model,
        'description': f'Inference ({description})',
        'results': results,
    }

    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
