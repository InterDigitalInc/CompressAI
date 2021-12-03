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

import argparse
import glob
import math
import os
import struct
import sys
import time

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

import compressai

from compressai.datasets import VideoSequenceInYUV420
from compressai.zoo import models

# from compressai.transforms.functional import rgb2ycbcr, yuv_444_to_420


torch.backends.cudnn.deterministic = True

model_ids = {k: i for i, k in enumerate(models.keys())}

metric_ids = {"mse": 0, "ms-ssim": 1}


def Average(lst):
    return sum(lst) / len(lst)


def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)
    return model_ids[model_name], code


def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4
    return (
        inverse_dict(model_ids)[model_id],
        inverse_dict(metric_ids)[metric],
        quality,
    )


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def _encode(image, model, metric, quality, coder, output):
    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    img = load_image(image)
    start = time.time()
    net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    load_time = time.time() - start

    x = img2torch(img)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    x = pad(x, p)

    with torch.no_grad():
        out = net.compress(x)

    shape = out["shape"]
    header = get_header(model, metric, quality)

    with Path(output).open("wb") as f:
        write_uchars(f, header)
        # write original image size
        write_uints(f, (h, w))
        # write shape and number of encoded latents
        write_uints(f, (shape[0], shape[1], len(out["strings"])))
        for s in out["strings"]:
            write_uints(f, (len(s[0]),))
            write_bytes(f, s[0])

    enc_time = time.time() - enc_start
    size = filesize(output)
    bpp = float(size) * 8 / (img.size[0] * img.size[1])
    print(
        f"{bpp:.3f} bpp |"
        f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
    )


# def _encode_video(image, model, metric, quality, coder, output):
def _encode_video(video, model, coder, output, device):
    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    start = time.time()
    #    net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    net = model.to(device).eval()
    load_time = time.time() - start

    video_sequence = VideoSequenceInYUV420(
        interval=1,
        src_addr=video["addr"],
        resolution=video["resolution"],
        num_frms=video["num_frms"],
    )
    lzs = int(math.log10(len(video_sequence)) + 1)

    frames_enc_time = []
    frames_coded_size = []
    frames_coded_bpp = []
    x_ref = None

    frmH, frmW = video_sequence.height, video_sequence.width

    p = 128
    for i in tqdm(range(len(video_sequence))):
        poc = str(i).zfill(lzs)
        frame_enc_start = time.time()

        frm = video_sequence[i]

        x = pad(frm.to(device), p)

        with torch.no_grad():
            if i == 0:
                x_out, out_info = net.encode_keyframe(x)
            else:
                x_out, out_info = net.encode_inter(x, x_ref)

        x_ref = x_out

        out_bin_file = f"{output}_poc{poc}.bin"

        shape = out_info["shape"]
        # header = get_header(model, metric, quality)

        with Path(out_bin_file).open("wb") as f:
            # write_uchars(f, header)
            # write original image size
            write_uints(f, (frmH, frmW))
            # write shape and number of encoded latents

            def write_body(f, shape, out_strings):
                write_uints(f, (shape[0], shape[1], len(out_strings)))
                for s in out_strings:
                    write_uints(f, (len(s[0]),))
                    write_bytes(f, s[0])

            if isinstance(shape, Dict):
                for shape, out in zip(
                    out_info["shape"].items(), out_info["strings"].items()
                ):
                    write_body(f, shape[1], out[1])
            else:
                write_body(f, shape, out_info["strings"])

        frame_enc_time = time.time() - frame_enc_start
        size = filesize(out_bin_file)
        bpp = float(size) * 8 / (frmH * frmW)

        frames_coded_bpp.append(bpp)
        frames_enc_time.append(frame_enc_time)
        frames_coded_size.append(size)

    enc_time = time.time() - enc_start

    print(
        "=== Encoding Summary ===\n"
        "------------------------\n"
        f" Total {i+1} Frames\n"
        f" Per frame encoding : {int(Average(frames_coded_size))} bytes (avg.) | "
        f" {Average(frames_coded_bpp):.3f} bpp (avg.) | "
        f" {Average(frames_enc_time):.3f}s (avg.) | "
        f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
    )


def _decode(inputpath, coder, show, output=None):
    compressai.set_entropy_coder(coder)

    dec_start = time.time()
    with Path(inputpath).open("rb") as f:
        model, metric, quality = parse_header(read_uchars(f, 2))
        original_size = read_uints(f, 2)
        shape = read_uints(f, 2)
        strings = []
        n_strings = read_uints(f, 1)[0]
        for _ in range(n_strings):
            s = read_bytes(f, read_uints(f, 1)[0])
            strings.append([s])

    print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")
    start = time.time()
    net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    load_time = time.time() - start

    with torch.no_grad():
        out = net.decompress(strings, shape)

    x_hat = crop(out["x_hat"], original_size)
    img = torch2img(x_hat)
    dec_time = time.time() - dec_start
    print(f"Decoded in {dec_time:.2f}s (model loading: {load_time:.2f}s)")

    if show:
        show_image(img)
    if output is not None:
        img.save(output)


# temp
def _decode_video(model, device, inputpath, coder, show, output=None):
    compressai.set_entropy_coder(coder)

    dec_start = time.time()

    all_coded_stream = glob.glob(f"{inputpath}/*.bin")
    all_coded_stream.sort(key=lambda x: int(x.split(".bin")[0].split("poc")[-1]))

    def read_body(f):
        lstrings = []
        shape = read_uints(f, 2)
        n_strings = read_uints(f, 1)[0]
        for _ in range(n_strings):
            s = read_bytes(f, read_uints(f, 1)[0])
            lstrings.append([s])

        return lstrings, shape

    frames_dec_time = []

    start = time.time()
    # net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    net = model.to(device).eval()
    load_time = time.time() - start

    x_ref = None
    for poc, bin_file in enumerate(tqdm(all_coded_stream)):
        frame_dec_start = time.time()

        with Path(bin_file).open("rb") as f:
            # model, metric, quality = parse_header(read_uchars(f, 2))
            original_size = read_uints(f, 2)

            if poc == 0:
                # print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")
                print("Model information will be printed out")

            with torch.no_grad():
                if poc == 0:
                    lstrings, shape = read_body(f)
                    out = net.decode_keyframe(lstrings, shape)
                else:
                    mstrings, mshape = read_body(f)
                    rstrings, rshape = read_body(f)
                    inter_strings = {"motion": mstrings, "residual": rstrings}
                    inter_shapes = {"motion": mshape, "residual": rshape}

                    out = net.decode_inter(x_ref, inter_strings, inter_shapes)

                x_ref = out

                x_hat = crop(out, original_size)
                # yuv = yuv_444_to_420(rgb2ycbcr(x_hat))
                img = torch2img(x_hat)

                if output is not None:
                    img.save(output)

                frame_dec_time = time.time() - frame_dec_start
                frames_dec_time.append(frame_dec_time)

                if show:
                    show_image(img)

    dec_time = time.time() - dec_start

    print(
        "=== Decoding Summary ===\n"
        "------------------------\n"
        f" Total {poc+1} Frames\n"
        f" Per frame decoding : {Average(frames_dec_time):.3f}s (avg.) | "
        f" Decoded in {dec_time:.2f}s (model loading: {load_time:.2f}s)"
    )


def show_image(img: Image.Image):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.title.set_text("Decoded image")
    ax.imshow(img)
    fig.tight_layout()
    plt.show()


def encode(argv):
    parser = argparse.ArgumentParser(description="Encode image to bit-stream")
    parser.add_argument("image", type=str)
    parser.add_argument(
        "--model",
        choices=models.keys(),
        default=list(models.keys())[0],
        help="NN model to use (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--metric",
        choices=metric_ids.keys(),
        default="mse",
        help="metric trained against (default: %(default)s",
    )
    parser.add_argument(
        "-q",
        "--quality",
        choices=list(range(1, 9)),
        type=int,
        default=3,
        help="Quality setting (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", help="Output path")
    args = parser.parse_args(argv)
    if not args.output:
        args.output = Path(Path(args.image).resolve().name).with_suffix(".bin")

    _encode(args.image, args.model, args.metric, args.quality, args.coder, args.output)


def video_encode(argv):
    print("Currently video encoder supports YUV input only")
    parser = argparse.ArgumentParser(description="Encode a video sequence to bitstream")
    parser.add_argument("video", type=str)
    parser.add_argument(
        "--video-resolution",
        type=int,
        nargs=2,
        default=(1080, 1920),
        help="resolution of the input video to code (default: %(default)s)",
    )
    parser.add_argument(
        "--num_frms",
        type=int,
        default=-1,
        help="Number of frames to be coded (default: %(default)s encode all frames)",
    )
    # temp
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    # parser.add_argument(
    #    "--model",
    #    choices=models.keys(),
    #    default=list(models.keys())[0],
    #    help="NN model to use (default: %(default)s)",
    # )
    # parser.add_argument(
    #    "-m",
    #    "--metric",
    #    choices=metric_ids.keys(),
    #    default="mse",
    #    help="metric trained against (default: %(default)s",
    # )
    # parser.add_argument(
    #    "-q",
    #    "--quality",
    #    choices=list(range(1, 9)),
    #    type=int,
    #    default=3,
    #    help="Quality setting (default: %(default)s)",
    # )
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    args = parser.parse_args(argv)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    file_name = Path(Path(args.video).resolve().name).stem
    output_prefix = f"{args.output}/coded_{file_name}"

    # temp code
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    from compressai.models.google import ScaleSpaceFlow

    model = ScaleSpaceFlow()

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Not clear this part
        model.load_state_dict(checkpoint["state_dict"])
        model.update(force=True)

    # _encode(args.video, args.model, args.metric, args.quality, args.coder, args.output)
    _encode_video(
        {
            "addr": args.video,
            "resolution": args.video_resolution,
            "num_frms": args.num_frms,
        },
        model,
        args.coder,
        output_prefix,
        device,
    )


def decode(argv):
    parser = argparse.ArgumentParser(description="Decode bit-stream to imager")
    parser.add_argument("input", type=str)
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-o", "--output", help="Output path")
    args = parser.parse_args(argv)
    _decode(args.input, args.coder, args.show, args.output)


def video_decode(argv):
    parser = argparse.ArgumentParser(description="Decode bit-stream to video sequence")
    parser.add_argument(
        "input", type=str, help="A directory includes coded bitstream per frame"
    )
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    # temp
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    args = parser.parse_args(argv)

    # temp codes
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    from compressai.models.google import ScaleSpaceFlow

    model = ScaleSpaceFlow()

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Not clear this part
        model.load_state_dict(checkpoint["state_dict"])
        model.update(force=True)

    _decode_video(model, device, args.input, args.coder, args.show, args.output)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "command", choices=["encode", "video_encode", "decode", "video_decode"]
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv[1:2])
    argv = argv[2:]
    torch.set_num_threads(1)  # just to be sure
    if args.command == "img_encode":
        encode(argv)
    elif args.command == "video_encode":
        video_encode(argv)
    elif args.command == "img_decode":
        decode(argv)
    elif args.command == "video_decode":
        video_decode(argv)


if __name__ == "__main__":
    main(sys.argv)
