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

from enum import Enum
from pathlib import Path
from typing import IO, Dict, NamedTuple, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch import Tensor
from torch.utils.model_zoo import tqdm
from torchvision.transforms import ToPILImage, ToTensor

import compressai

from compressai.datasets import RawVideoSequence, VideoFormat
from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)
from compressai.zoo import image_models, models

torch.backends.cudnn.deterministic = True

model_ids = {k: i for i, k in enumerate(models.keys())}

metric_ids = {"mse": 0, "ms-ssim": 1}

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]


class CodecType(Enum):
    IMAGE_CODEC = 0
    VIDEO_CODEC = 1
    NUM_CODEC_TYPE = 2


class CodecInfo(NamedTuple):
    codec_header: Tuple
    original_size: Tuple
    original_bitdepth: int
    net: Dict
    device: str


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


def get_header(model_name, metric, quality, num_of_frames, codec_type: Enum):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    - 4 bytes for number of frames to be coded (only applicable for video)
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)

    if codec_type == CodecType.VIDEO_CODEC:
        return model_ids[model_name], code, num_of_frames

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


def to_tensors(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_value: int = 1,
    device: str = "cpu",
) -> Frame:
    return tuple(
        torch.from_numpy(np.true_divide(c, max_value, dtype=np.float32)).to(device)
        for c in frame
    )


def convert_yuv420_rgb(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device, max_val: int
) -> Tensor:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    frame = to_tensors(frame, device=str(device), max_value=max_val)
    frame = yuv_420_to_444(
        tuple(c.unsqueeze(0).unsqueeze(0) for c in frame), mode="bicubic"  # type: ignore
    )
    return ycbcr2rgb(frame)  # type: ignore


def convert_rgb_yuv420(frame: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    return yuv_444_to_420(rgb2ycbcr(frame), mode="avg_pool")


def pad(x, p=2**6):
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


def convert_output(t: Tensor, bitdepth: int = 8) -> np.array:
    assert bitdepth in (8, 10)
    # [0,1] fp ->  [0, 2**bitstream-1] uint
    dtype = np.uint8 if bitdepth == 8 else np.uint16
    t = (t.clamp(0, 1) * (2**bitdepth - 1)).cpu().squeeze()
    arr = t.numpy().astype(dtype)
    return arr


def write_frame(fout: IO[bytes], frame: Frame, bitdepth: np.uint = 8):
    for plane in frame:
        convert_output(plane, bitdepth).tofile(fout)


def encode_image(input, codec: CodecInfo, output):
    if Path(input).suffix == ".yuv":
        # encode first frame of YUV sequence only
        org_seq = RawVideoSequence.from_file(input)
        bitdepth = org_seq.bitdepth
        max_val = 2**bitdepth - 1
        if org_seq.format != VideoFormat.YUV420:
            raise NotImplementedError(f"Unsupported video format: {org_seq.format}")
        x = convert_yuv420_rgb(org_seq[0], codec.device, max_val)
    else:
        img = load_image(input)
        x = img2torch(img)
        bitdepth = 8

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    x = pad(x, p)

    with torch.no_grad():
        out = codec.net.compress(x)

    shape = out["shape"]

    with Path(output).open("wb") as f:
        write_uchars(f, codec.codec_header)
        # write original image size
        write_uints(f, (h, w))
        # write original bitdepth
        write_uchars(f, (bitdepth,))
        # write shape and number of encoded latents
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (h * w)

    return {"bpp": bpp}


def encode_video(input, codec: CodecInfo, output):
    if Path(input).suffix != ".yuv":
        raise NotImplementedError(
            f"Unsupported video file extension: {Path(input).suffix}"
        )

    # encode frames of YUV sequence only
    org_seq = RawVideoSequence.from_file(input)
    bitdepth = org_seq.bitdepth
    max_val = 2**bitdepth - 1
    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    num_frames = codec.codec_header[2]
    if num_frames < 0:
        num_frames = org_seq.total_frms

    avg_frame_enc_time = []

    f = Path(output).open("wb")
    with torch.no_grad():
        # Write Video Header
        write_uchars(f, codec.codec_header[0:2])
        # write original image size
        write_uints(f, (org_seq.height, org_seq.width))
        # write original bitdepth
        write_uchars(f, (bitdepth,))
        # write number of coded frames
        write_uints(f, (num_frames,))

        x_ref = None
        with tqdm(total=num_frames) as pbar:
            for i in range(num_frames):
                frm_enc_start = time.time()

                x_cur = convert_yuv420_rgb(org_seq[i], codec.device, max_val)
                h, w = x_cur.size(2), x_cur.size(3)
                p = 128  # maximum 7 strides of 2
                x_cur = pad(x_cur, p)

                if i == 0:
                    x_out, out_info = codec.net.encode_keyframe(x_cur)
                    write_body(f, out_info["shape"], out_info["strings"])
                else:
                    x_out, out_info = codec.net.encode_inter(x_cur, x_ref)
                    for shape, out in zip(
                        out_info["shape"].items(), out_info["strings"].items()
                    ):
                        write_body(f, shape[1], out[1])

                x_ref = x_out.clamp(0, 1)

                avg_frame_enc_time.append((time.time() - frm_enc_start))

                pbar.update(1)

        org_seq.close()
    f.close()

    size = filesize(output)
    bpp = float(size) * 8 / (h * w * num_frames)

    return {"bpp": bpp, "avg_frm_enc_time": np.mean(avg_frame_enc_time)}


def _encode(input, num_of_frames, model, metric, quality, coder, device, output):
    encode_func = {
        CodecType.IMAGE_CODEC: encode_image,
        CodecType.VIDEO_CODEC: encode_video,
    }

    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    start = time.time()
    model_info = models[model]
    net = model_info(quality=quality, metric=metric, pretrained=True).to(device).eval()
    codec_type = (
        CodecType.IMAGE_CODEC if model in image_models else CodecType.VIDEO_CODEC
    )

    codec_header_info = get_header(model, metric, quality, num_of_frames, codec_type)
    load_time = time.time() - start

    if not Path(input).is_file():
        raise FileNotFoundError(f"{input} does not exist")

    codec_info = CodecInfo(codec_header_info, None, None, net, device)
    out = encode_func[codec_type](input, codec_info, output)

    enc_time = time.time() - enc_start

    print(
        f"{out['bpp']:.3f} bpp |"
        f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
    )


def decode_image(f, codec: CodecInfo, output):
    strings, shape = read_body(f)
    with torch.no_grad():
        out = codec.net.decompress(strings, shape)

    x_hat = crop(out["x_hat"], codec.original_size)

    img = torch2img(x_hat)

    if output is not None:
        if Path(output).suffix == ".yuv":
            rec = convert_rgb_yuv420(x_hat)
            with Path(output).open("wb") as fout:
                write_frame(fout, rec, codec.original_bitdepth)
        else:
            img.save(output)

    return {"img": img}


def decode_video(f, codec: CodecInfo, output):
    # read number of coded frames
    num_frames = read_uints(f, 1)[0]

    avg_frame_dec_time = []

    with torch.no_grad():
        x_ref = None
        with tqdm(total=num_frames) as pbar:
            for i in range(num_frames):
                frm_dec_start = time.time()

                if i == 0:
                    strings, shape = read_body(f)
                    x_out = codec.net.decode_keyframe(strings, shape)
                else:
                    mstrings, mshape = read_body(f)
                    rstrings, rshape = read_body(f)
                    inter_strings = {"motion": mstrings, "residual": rstrings}
                    inter_shapes = {"motion": mshape, "residual": rshape}

                    x_out = codec.net.decode_inter(x_ref, inter_strings, inter_shapes)

                x_ref = x_out.clamp(0, 1)

                avg_frame_dec_time.append((time.time() - frm_dec_start))

                x_hat = crop(x_out, codec.original_size)
                img = torch2img(x_hat)

                if output is not None:
                    if Path(output).suffix == ".yuv":
                        rec = convert_rgb_yuv420(x_hat)
                        wopt = "wb" if i == 0 else "ab"
                        with Path(output).open(wopt) as fout:
                            write_frame(fout, rec, codec.original_bitdepth)
                    else:
                        img.save(output)

                pbar.update(1)

    return {"img": img, "avg_frm_dec_time": np.mean(avg_frame_dec_time)}


def _decode(inputpath, coder, show, device, output=None):
    decode_func = {
        CodecType.IMAGE_CODEC: decode_image,
        CodecType.VIDEO_CODEC: decode_video,
    }

    compressai.set_entropy_coder(coder)

    dec_start = time.time()
    with Path(inputpath).open("rb") as f:
        model, metric, quality = parse_header(read_uchars(f, 2))

        original_size = read_uints(f, 2)
        original_bitdepth = read_uchars(f, 1)[0]

        start = time.time()
        model_info = models[model]
        net = (
            model_info(quality=quality, metric=metric, pretrained=True)
            .to(device)
            .eval()
        )
        codec_type = (
            CodecType.IMAGE_CODEC if model in image_models else CodecType.VIDEO_CODEC
        )

        load_time = time.time() - start
        print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")

        stream_info = CodecInfo(None, original_size, original_bitdepth, net, device)
        out = decode_func[codec_type](f, stream_info, output)

    dec_time = time.time() - dec_start
    print(f"Decoded in {dec_time:.2f}s (model loading: {load_time:.2f}s)")

    if show:
        # For video, only the last frame is shown
        show_image(out["img"])


def show_image(img: Image.Image):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.title.set_text("Decoded image")
    ax.imshow(img)
    fig.tight_layout()
    plt.show()


def encode(argv):
    parser = argparse.ArgumentParser(description="Encode image/video to bit-stream")
    parser.add_argument(
        "input",
        type=str,
        help="Input path, the first frame will be encoded with a NN image codec if the input is a raw yuv sequence",
    )
    parser.add_argument(
        "-f",
        "--num_of_frames",
        default=-1,
        type=int,
        help="Number of frames to be coded. -1 will encode all frames of input (default: %(default)s)",
    )
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
        help="metric trained against (default: %(default)s)",
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
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    args = parser.parse_args(argv)
    if not args.output:
        args.output = Path(Path(args.input).resolve().name).with_suffix(".bin")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    _encode(
        args.input,
        args.num_of_frames,
        args.model,
        args.metric,
        args.quality,
        args.coder,
        device,
        args.output,
    )


def decode(argv):
    parser = argparse.ArgumentParser(description="Decode bit-stream to image/video")
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
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    args = parser.parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    _decode(args.input, args.coder, args.show, device, args.output)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("command", choices=["encode", "decode"])
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv[0:1])
    argv = argv[1:]
    torch.set_num_threads(1)  # just to be sure
    if args.command == "encode":
        encode(argv)
    elif args.command == "decode":
        decode(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
