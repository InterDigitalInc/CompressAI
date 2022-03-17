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
import json
import math
import struct
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm

import compressai

from compressai.datasets import RawVideoSequence, VideoFormat
from compressai.models.video.google import ScaleSpaceFlow
from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)
from compressai.zoo import video_models as pretrained_models

models = {"ssf2020": ScaleSpaceFlow}

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]

RAWVIDEO_EXTENSIONS = (".yuv",)  # read raw yuv videos for now


def collect_videos(rootpath: str) -> List[str]:
    video_files = []
    for ext in RAWVIDEO_EXTENSIONS:
        video_files.extend(Path(rootpath).glob(f"*{ext}"))
    return video_files


# TODO (racapef) duplicate from bench
def to_tensors(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_value: int = 1,
    device: str = "cpu",
) -> Frame:
    return tuple(
        torch.from_numpy(np.true_divide(c, max_value, dtype=np.float32)).to(device)
        for c in frame
    )


def aggregate_results(filepaths: List[Path]) -> Dict[str, Any]:
    metrics = defaultdict(list)

    # sum
    for f in filepaths:
        with f.open("r") as fd:
            data = json.load(fd)
        for k, v in data["results"].items():
            metrics[k].append(v)

    # normalize
    agg = {k: np.mean(v) for k, v in metrics.items()}
    return agg


def convert_yuv420_to_rgb(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device, max_val: int
) -> Tensor:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    out = to_tensors(frame, device=str(device), max_value=max_val)
    out = yuv_420_to_444(
        tuple(c.unsqueeze(0).unsqueeze(0) for c in out), mode="bicubic"  # type: ignore
    )
    return ycbcr2rgb(out)  # type: ignore


def convert_rgb_to_yuv420(frame: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    return yuv_444_to_420(rgb2ycbcr(frame), mode="avg_pool")


def pad(x: Tensor, p: int = 2 ** (4 + 3)) -> Tuple[Tensor, Tuple[int, ...]]:
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    x = F.pad(
        x,
        padding,
        mode="constant",
        value=0,
    )
    return x, padding


def crop(x: Tensor, padding: Tuple[int, ...]) -> Tensor:
    return F.pad(x, tuple(-p for p in padding))


def compute_metrics_for_frame(
    org_frame: Frame,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # YCbCr metrics
    org_yuv = to_tensors(org_frame, device=str(device), max_value=max_val)
    org_yuv = tuple(p.unsqueeze(0).unsqueeze(0) for p in org_yuv)  # type: ignore
    rec_yuv = convert_rgb_to_yuv420(rec_frame)
    for i, component in enumerate("yuv"):
        org = (org_yuv[i] * max_val).clamp(0, max_val).round()
        rec = (rec_yuv[i] * max_val).clamp(0, max_val).round()
        out[f"psnr-{component}"] = 20 * np.log10(max_val) - 10 * torch.log10(
            (org - rec).pow(2).mean()
        )
    out["psnr-yuv"] = (4 * out["psnr-y"] + out["psnr-u"] + out["psnr-v"]) / 6

    # RGB metrics
    org_rgb = convert_yuv420_to_rgb(
        org_frame, device, max_val
    )  # ycbcr2rgb(yuv_420_to_444(org_frame, mode="bicubic"))  # type: ignore
    org_rgb = (org_rgb * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_rgb - rec_frame).pow(2).mean()
    psnr_rgb = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)

    ms_ssim_rgb = ms_ssim(org_rgb, rec_frame, data_range=max_val)
    out.update({"ms-ssim-rgb": ms_ssim_rgb, "mse-rgb": mse_rgb, "psnr-rgb": psnr_rgb})

    return out


def estimate_bits_frame(likelihoods) -> float:
    bpp = sum(
        (torch.log(lkl[k]).sum() / (-math.log(2)))
        for lkl in likelihoods.values()
        for k in ("y", "z")
    )
    return bpp


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


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


@torch.no_grad()
def eval_model(
    net: nn.Module, sequence: Path, binpath: Path, keep_binaries: bool = False
) -> Dict[str, Any]:
    org_seq = RawVideoSequence.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(net.parameters()).device
    num_frames = len(org_seq)
    max_val = 2**org_seq.bitdepth - 1
    results = defaultdict(list)

    f = binpath.open("wb")

    print(f" encoding {sequence.stem}", file=sys.stderr)
    # write original image size
    write_uints(f, (org_seq.height, org_seq.width))
    # write original bitdepth
    write_uchars(f, (org_seq.bitdepth,))
    # write number of coded frames
    write_uints(f, (num_frames,))
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            x_cur = convert_yuv420_to_rgb(org_seq[i], device, max_val)
            x_cur, padding = pad(x_cur)

            if i == 0:
                x_rec, enc_info = net.encode_keyframe(x_cur)
                write_body(f, enc_info["shape"], enc_info["strings"])
                # x_rec = net.decode_keyframe(enc_info["strings"], enc_info["shape"])
            else:
                x_rec, enc_info = net.encode_inter(x_cur, x_rec)
                for shape, out in zip(
                    enc_info["shape"].items(), enc_info["strings"].items()
                ):
                    write_body(f, shape[1], out[1])
                # x_rec = net.decode_inter(x_rec, enc_info["strings"], enc_info["shape"])

            x_rec = x_rec.clamp(0, 1)
            metrics = compute_metrics_for_frame(
                org_seq[i],
                crop(x_rec, padding),
                device,
                max_val,
            )

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)
    f.close()

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }

    seq_results["bitrate"] = (
        float(filesize(binpath)) * 8 * org_seq.framerate / (num_frames * 1000)
    )

    if not keep_binaries:
        binpath.unlink()

    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results


@torch.no_grad()
def eval_model_entropy_estimation(net: nn.Module, sequence: Path) -> Dict[str, Any]:
    org_seq = RawVideoSequence.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(net.parameters()).device
    num_frames = len(org_seq)
    max_val = 2**org_seq.bitdepth - 1
    results = defaultdict(list)
    print(f" encoding {sequence.stem}", file=sys.stderr)
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            x_cur = convert_yuv420_to_rgb(org_seq[i], device, max_val)
            x_cur, padding = pad(x_cur)

            if i == 0:
                x_rec, likelihoods = net.forward_keyframe(x_cur)  # type:ignore
            else:
                x_rec, likelihoods = net.forward_inter(x_cur, x_rec)  # type:ignore

            x_rec = x_rec.clamp(0, 1)

            metrics = compute_metrics_for_frame(
                org_seq[i],
                crop(x_rec, padding),
                device,
                max_val,
            )
            metrics["bitrate"] = estimate_bits_frame(likelihoods)

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }
    seq_results["bitrate"] = float(seq_results["bitrate"]) * org_seq.framerate / 1000
    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results


def run_inference(
    filepaths,
    net: nn.Module,
    outputdir: Path,
    force: bool = False,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any,
) -> Dict[str, Any]:
    results_paths = []

    for filepath in filepaths:
        sequence_metrics_path = Path(outputdir) / f"{filepath.stem}-{trained_net}.json"
        results_paths.append(sequence_metrics_path)

        if force:
            sequence_metrics_path.unlink(missing_ok=True)
        if sequence_metrics_path.is_file():
            continue

        with amp.autocast(enabled=args["half"]):
            with torch.no_grad():
                if entropy_estimation:
                    metrics = eval_model_entropy_estimation(net, filepath)
                else:
                    sequence_bin = sequence_metrics_path.with_suffix(".bin")
                    metrics = eval_model(
                        net, filepath, sequence_bin, args["keep_binaries"]
                    )
        with sequence_metrics_path.open("wb") as f:
            output = {
                "source": filepath.stem,
                "name": args["architecture"],
                "description": f"Inference ({description})",
                "results": metrics,
            }
            f.write(json.dumps(output, indent=2).encode())
    results = aggregate_results(results_paths)
    return results


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    state_dict = torch.load(checkpoint_path)
    state_dict = state_dict.get("network", state_dict)
    net = models[arch]()
    net.load_state_dict(state_dict)
    net.eval()
    return net


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video compression network evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="sequences directory")
    parent_parser.add_argument("output", type=str, help="output directory")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite previous runs"
    )
    parent_parser.add_argument("--cuda", action="store_true", help="use cuda")
    parent_parser.add_argument("--half", action="store_true", help="use AMP")
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--keep_binaries",
        action="store_true",
        help="keep bitstream files in output directory",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(help="model source", dest="source")
    subparsers.required = True

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )
    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    return parser


def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)

    if not args.source:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)
    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )
    filepaths = collect_videos(args.dataset)
    if len(filepaths) == 0:
        print("Error: no video found in directory.", file=sys.stderr)
        raise SystemExit(1)

    # create output directory
    outputdir = args.output
    Path(outputdir).mkdir(parents=True, exist_ok=True)

    if args.source == "pretrained":
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    else:
        runs = args.paths
        opts = (args.architecture,)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        model = load_func(*opts, run)
        if args.source == "pretrained":
            trained_net = f"{args.architecture}-{args.metric}-{run}-{description}"
        else:
            cpt_name = Path(run).name[: -len(".tar.pth")]  # removesuffix() python3.9
            trained_net = f"{cpt_name}-{description}"
        print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
            if args.half:
                model = model.half()
        args_dict = vars(args)
        metrics = run_inference(
            filepaths,
            model,
            outputdir,
            trained_net=trained_net,
            description=description,
            **args_dict,
        )
        results["q"].append(trained_net)
        for k, v in metrics.items():
            results[k].append(v)

    output = {
        "name": f"{args.architecture}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }

    with (Path(f"{outputdir}/{args.architecture}-{description}.json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
