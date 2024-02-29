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
import sys

import torch


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Extract the decoder from the full model and convert it into ONNX. Extract decoder info to a pickle."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="model", help="output file prefix"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument(
        "--extract_info",
        action="store_true",
        help="extract additional infos in the model",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to a pth model, not a state_dict"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    if False:  # torch.cuda.is_available():
        device = torch.device("cuda:{}".format(torch.cuda.current_device()))
        print("[INFO] Using device:", torch.cuda.get_device_name(device))
        loc = "cuda"
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
        loc = "cpu"
    args = parse_args(argv)

    print("[INFO] read checkpoint: ", args.model)
    checkpoint = torch.load(args.model, map_location=loc)
    autoencoder = checkpoint["model"]

    if not hasattr(autoencoder, "g_s"):
        quit("[ERROR] no decoder in the model")
    decoder = autoencoder.g_s

    if args.verbose:
        for i, _ in checkpoint.items():
            print(i)
        print("[INFO] model:", autoencoder)
        print("[INFO] decoder:", decoder)

    filename = args.output
    decoder.eval()
    # decoder = decoder.to(memory_format=torch.channels_last)
    print("[INFO] save decoder : ", filename + "_dec.onnx")
    dummy_input = torch.randn(1, decoder[0].in_channels, 8, 8, requires_grad=True)
    if args.verbose:
        print("[INFO] size input dummy", dummy_input.size())
    dummy_input = dummy_input.to(device)
    torch.onnx.export(decoder, dummy_input, filename + "_dec.onnx", opset_version=11)

    if True:
        if not hasattr(autoencoder, "g_a"):
            quit("[ERROR] no encoder in the model")
        encoder = autoencoder.g_a
        print("[INFO] save encoder : ", filename + "_enc.onnx")
        dummy_input = torch.randn(1, encoder[0].in_channels, 8, 8, requires_grad=True)
        torch.onnx.export(
            encoder, dummy_input, filename + "_enc.onnx", opset_version=11
        )

    if args.extract_info:
        cdf = checkpoint["cdf"]
        cdflen = checkpoint["cdflen"]
        cdfoff = checkpoint["cdfoff"]
        quant_layer = checkpoint["quant_layer"]

        dict_quantizers = {}
        i = 0
        for name in decoder.named_parameters():
            dict_quantizers[name[0]] = quant_layer[i]
            i = i + 1

        if args.verbose:
            print("[INFO] quantizers: ", dict_quantizers)

        dict_cdf = {}
        dict_cdf["cdf"] = cdf.cpu().detach().numpy()
        dict_cdf["cdflen"] = cdflen.cpu().detach().numpy()
        dict_cdf["cdfoff"] = cdfoff.cpu().detach().numpy()
        dict_cdf["quantizers"] = dict_quantizers

        if args.verbose:
            print("[INFO] dec_info: ", dict_cdf)

        pickle.dump(dict_cdf, open(filename + "_info.pkl", "wb"))
        print("[INFO] wrote decoder info: ", filename + "_info.pkl")


if __name__ == "__main__":
    main(sys.argv[1:])
