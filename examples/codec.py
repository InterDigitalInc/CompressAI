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

import argparse
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


def BoolConvert(a):
    b = [False, True]
    return b[int(a)]


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
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


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
    return len(values) * 1


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


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


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
        write_body(f, shape, out["strings"])

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

    frames_enc_time = []
    frames_coded_size = []
    frames_coded_bpp = []
    x_ref = None

    frmH, frmW = video_sequence.height, video_sequence.width

    p = 128
    with Path(output).open("wb") as fout:
        is_first_frm = False

        # header = get_header(model, metric, quality)
        # write_uchars(f, header)
        # write original image size
        write_uints(fout, (frmH, frmW))

        for i in tqdm(range(len(video_sequence))):
            is_first_frm = True if i == 0 else False
            is_last_frm = True if (i + 1) == len(video_sequence) else False
            frm_bytes = 0
            frm_enc_start = time.time()

            frm = video_sequence[i]

            x = pad(frm.to(device), p)

            with torch.no_grad():
                if is_first_frm:
                    x_out, out_info = net.encode_keyframe(x)
                    shape = out_info["shape"]

                    # write a frame number
                    frm_bytes = write_uints(fout, (i,))
                    # write shape and number of encoded latents
                    frm_bytes += write_body(fout, shape, out_info["strings"])
                else:
                    x_out, out_info = net.encode_inter(x, x_ref)
                    shape = out_info["shape"]

                    assert isinstance(shape, Dict) is True

                    # write a frame number
                    frm_bytes = write_uints(fout, (i,))
                    # write shape and number of encoded latents for motion and residuals
                    for shape, out in zip(
                        out_info["shape"].items(), out_info["strings"].items()
                    ):
                        frm_bytes += write_body(fout, shape[1], out[1])

                frm_bytes += write_uchars(fout, (is_last_frm,))

            x_ref = x_out

            frm_enc_time = time.time() - frm_enc_start
            bpp = float(frm_bytes) * 8 / (frmH * frmW)

            frames_coded_bpp.append(bpp)
            frames_enc_time.append(frm_enc_time)
            frames_coded_size.append(frm_bytes)

        video_sequence.close()

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
        strings, shape = read_body(f)

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

    frames_dec_time = []

    start = time.time()
    # net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    net = model.to(device).eval()
    load_time = time.time() - start

    x_ref = None

    with Path(inputpath).open("rb") as fin:
        first_frm_done = False
        is_last_frm = False

        # model, metric, quality = parse_header(read_uchars(f, 2))
        original_size = read_uints(fin, 2)

        # print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")
        print("Model information will be printed out")

        while not is_last_frm:
            # for poc, bin_file in enumerate(tqdm(all_coded_stream)):
            frame_dec_start = time.time()

            with torch.no_grad():
                if not first_frm_done:
                    # write a frame number
                    poc = read_uints(fin, 1)[0]

                    assert poc == 0
                    lstrings, shape = read_body(fin)
                    out = net.decode_keyframe(lstrings, shape)
                    first_frm_done = True
                else:
                    # write a frame number
                    poc = read_uints(fin, 1)[0]
                    assert poc > 0

                    mstrings, mshape = read_body(fin)
                    rstrings, rshape = read_body(fin)
                    inter_strings = {"motion": mstrings, "residual": rstrings}
                    inter_shapes = {"motion": mshape, "residual": rshape}

                    out = net.decode_inter(x_ref, inter_strings, inter_shapes)

                is_last_frm = BoolConvert(read_uchars(fin, 1)[0])

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
    parser.add_argument("-o", "--output", help="Output path")
    args = parser.parse_args(argv)
    if not args.output:
        args.output = Path(Path(args.image).resolve().name).with_suffix(".bin")

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
        args.output,
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
    parser.add_argument("input", type=str)
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
