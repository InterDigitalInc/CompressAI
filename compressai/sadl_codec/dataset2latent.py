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
import sys

from os.path import exists

import numpy as np
import torch

max_val_abs = 32767
max_proba = 65536
min_cdf_bound = 2


def parse_args(argv):
    parser = argparse.ArgumentParser(description="extract latent from dataset")
    parser.add_argument("--model", type=str, required=True, help="model pth")
    parser.add_argument("--input", type=str, required=True, help="npy batch")
    parser.add_argument(
        "--nb_max", type=int, help="max latent to generate (all if not set)"
    )
    parser.add_argument(
        "--float_NHWC",
        action="store_true",
        help="store in float format SADL compatible",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="tensor in binary format int16"
    )
    parser.add_argument("--verbose", action="store_true", help="verbose")
    args = parser.parse_args(argv)
    return args


def main(argv):  # noqa: C901
    args = parse_args(argv)

    print("[INFO] read checkpoint: ", args.model)
    checkpoint = torch.load(args.model, map_location="cpu")

    if args.verbose:
        for i, _k in checkpoint.items():
            print(i)

    autoencoder = checkpoint["model"]

    encoder = autoencoder.g_a

    if args.verbose:
        print("[INFO] encoder: {}".format(encoder))

    block_size = [256, 256]

    data: np.ndarray = np.memmap(args.input, mode="r", dtype="uint8")
    data = data.reshape((-1, block_size[0], block_size[1], 3))
    print("[INFO] training shape={}".format(data.shape))
    # get info with one run
    x1 = np.transpose(data[0], (2, 1, 0)).copy().astype("float32") / 255.0
    x1 = torch.tensor(x1)
    y_encode = encoder(x1.unsqueeze(0))
    y_encode = torch.round(y_encode)
    y_encode_int = y_encode.int()
    C = y_encode_int.shape[1]
    H = y_encode_int.shape[2]
    W = y_encode_int.shape[3]
    N = data.shape[0]

    if args.nb_max is not None:
        N = args.nb_max

    if args.float_NHWC:
        outputs: np.ndarray = np.memmap(
            args.output, mode="w+", dtype="float32", shape=(N, H, W, C)
        )
    else:
        outputs: np.ndarray = np.memmap(
            args.output, mode="w+", dtype="int16", shape=(N, C, H, W)
        )

    if exists(args.output + ".means"):
        m = np.fromfile(args.output + ".means", dtype="float32")
    else:
        m = np.zeros(C, dtype=float)
        for i in range(N):
            x1 = np.transpose(data[i], (2, 1, 0)).copy().astype("float32") / 255.0
            x1 = torch.tensor(x1)
            y_encode = encoder(x1.unsqueeze(0))
            m += np.mean(y_encode.cpu().detach().numpy(), (0, 2, 3))
        m /= N

    m = m.reshape(1, C, 1, 1)
    m_tensor = np.ones((1, C, H, W), dtype="float32") * m

    for i in range(N):
        x1 = np.transpose(data[i], (2, 1, 0)).copy().astype("float32") / 255.0
        x1 = torch.tensor(x1)
        y_encode = encoder(x1.unsqueeze(0))
        y_encode = np.round(y_encode.cpu().detach().numpy() - m_tensor)
        if args.float_NHWC:
            outputs[i] = np.transpose(y_encode.astype("float32"), (0, 2, 3, 1))
        else:
            # y_encode_int = y_encode.int()
            outputs[i] = y_encode.astype("int16")
        if args.verbose:
            sys.stdout.write("\r{}%     ".format(int(100.0 * i / N)))
            sys.stdout.flush()

    print("[INFO] wrote {}".format(args.output))
    if not exists(args.output + ".means"):
        m.astype("float32").tofile(args.output + ".means")


if __name__ == "__main__":
    main(sys.argv[1:])
