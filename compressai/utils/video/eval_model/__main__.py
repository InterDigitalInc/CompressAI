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
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm

from compressai.datasets import RawVideoSequence, VideoFormat
from compressai.models.video.google import ScaleSpaceFlow
from compressai.transforms.functional import ycbcr2rgb, yuv_420_to_444
from compressai.zoo import models as pretrained_models

models = {"ssf2020": ScaleSpaceFlow}

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]


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


# TODO (racapef) duplicate from bench
def aggregate_results(filepaths: List[Path]) -> Dict[str, Any]:
    metrics = defaultdict(list)

    # sum
    for f in filepaths:
        with f.open("r") as fd:
            data = json.load(fd)
        for k, v in data.items():
            metrics[k].append(v)

    # normalize
    agg = {k: np.mean(v) for k, v in metrics.items()}
    return agg


# TODO (racapef) duplicate from bench
def convert_legal_to_full_range(frame: Frame, bitdepth: int = 8) -> Frame:
    ranges = torch.tensor([16, 235, 16, 240], device=frame[0].device)
    ranges *= 2 ** (bitdepth - 8)
    ymin, ymax, cmin, cmax = ranges
    y, cb, cr = frame

    y = (y - ymin) / (ymax - ymin)
    cb = (cb - cmin) / (cmax - cmin)
    cr = (cr - cmin) / (cmax - cmin)

    return y, cb, cr


def convert_frame(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    bitdepth: int,
) -> Tensor:
    # yuv420 [0, 2**bitdepth-1] (legal) to rgb 444 [0, 1] only for now
    out = to_tensors(frame, device=str(device))
    out = convert_legal_to_full_range(out, bitdepth)
    out = yuv_420_to_444(
        tuple(c.unsqueeze(0).unsqueeze(0) for c in out), mode="bicubic"  # type: ignore
    )
    return ycbcr2rgb(out).clamp(0, 1)  # type: ignore


def pad(x: Tensor, p: int = 2 ** (4 + 3)) -> Tuple[Tensor, Tuple[int, ...]]:
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    x = F.pad(x, padding, mode="replicate")
    return x, padding


def crop(x: Tensor, padding: Tuple[int, ...]) -> Tensor:
    return F.pad(x, tuple(-p for p in padding))


def compute_metrics_for_frame(
    org: Tensor, rec: Tensor, bitdepth: int
) -> Dict[str, Any]:
    max_val = 2 ** bitdepth - 1
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    mse = (org - rec).pow(2).mean()
    return {
        "mse": mse,
        "rgb_psnr": 20 * np.log10(max_val) - 10 * torch.log10(mse),
    }


def compute_bpp(likelihoods, num_pixels: int) -> float:
    bpp = sum(
        (torch.log(lkl[k]).sum() / (-math.log(2) * num_pixels))
        for lkl in likelihoods.values()
        for k in ("y", "z")
    )
    return bpp


def eval_model(net: nn.Module, sequence: Path) -> Dict[str, Any]:
    org_seq = RawVideoSequence.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(net.parameters()).device
    num_frames = len(org_seq)
    num_pixels = org_seq.height * org_seq.width
    results = defaultdict(list)
    # max_val = 2 ** org_seq.bitdepth - 1

    # output = Path(sequence.name).open("wb")
    with tqdm(total=num_frames) as pbar:
        # rec_frame_ai = None
        for i in range(num_frames):
            cur_frame = convert_frame(org_seq[i], device, org_seq.bitdepth)
            cur_frame, padding = pad(cur_frame)

            if i == 0:
                rec_frame, likelihoods = net.forward_keyframe(cur_frame)  # type:ignore
            else:
                rec_frame, likelihoods = net.forward_inter(
                    cur_frame, rec_frame
                )  # type:ignore

                # rec_frame_ai, likelihoods_ai = net.forward_keyframe(
                #     cur_frame
                # )  # type:ignore
                # rec_frame_ai = rec_frame_ai.clamp(0, 1)
                # metrics_ai = compute_metrics_for_frame(
                #     crop(cur_frame, padding),
                #     crop(rec_frame_ai, padding),
                #     org_seq.bitdepth,
                # )
                # metrics_ai["bpp"] = compute_bpp(likelihoods_ai, num_pixels)

            rec_frame = rec_frame.clamp(0, 1)

            metrics = compute_metrics_for_frame(
                crop(cur_frame, padding), crop(rec_frame, padding), org_seq.bitdepth
            )
            metrics["bpp"] = compute_bpp(likelihoods, num_pixels)

            # if (
            #     rec_frame_ai is not None
            #     and metrics_ai["rgb_psnr"] > metrics["rgb_psnr"]
            #     and metrics_ai["bpp"] <= metrics["bpp"]
            # ):
            #     print(f"INTRASWITCH frame_id={i} | "
            #           f"{metrics_ai['rgb_psnr']:.2f} {metrics_ai['bpp']:.3f} | "
            #           f"{metrics['rgb_psnr']:.2f} {metrics['bpp']:.3f}"
            #           )
            #     metrics = metrics_ai
            #     rec_frame = rec_frame_ai

            # TODO: per component metrics
            # out = yuv_444_to_420(rgb2ycbcr(crop(rec_frame, padding).clamp(0, 1)))
            # org_frame_fr = convert_legal_to_full_range(
            #     to_tensors(org_seq[i], device=str(device)), org_seq.bitdepth
            # )
            # for c, n in enumerate("yuv"):
            #     metrics[f"{n}_mse"] = (
            #         (
            #             org_frame_fr[c].mul(max_val).clamp(0, max_val).round()
            #             - out[c].mul(max_val).clamp(0, max_val).round()
            #         )
            #         .pow(2)
            #         .mean()
            #     )

            # TODO: option to save rec YUV
            # for c in out:
            #     output.write(c.cpu().squeeze().mul(255.0).byte().numpy().tobytes())
            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    # compute average metrics for sequence
    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }
    # filesize = get_filesize(bitstream_path)
    # seq_results["bpp"] = 8.0 * filesize / (num_frames * org_seq.width * org_seq.height)
    # TODO: per component metrics
    # for component in "yuv":
    #     seq_results[f"{component}_psnr"] = (
    #         20 * np.log10(max_val)
    #         - 10 * torch.log10(seq_results.pop(f"{component}_mse")).item()
    #     )
    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results


def run_inference(
    dataset: Path,
    net: nn.Module,
    outputdir: Path,
    force: bool = False,
    dry_run: bool = False,
    entropy_estimation: bool = False,
    **args: Any,
) -> Dict[str, Any]:
    # create output directory
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    results_paths = []

    for filepath in sorted(Path(dataset).glob("*.yuv")):
        outputpath = Path(outputdir) / filepath.with_suffix(".bin").name
        sequence_metrics_path = outputpath.with_suffix(".json")
        results_paths.append(sequence_metrics_path)

        if force:
            sequence_metrics_path.unlink(missing_ok=True)
        if sequence_metrics_path.is_file():
            continue

        if not entropy_estimation:
            raise NotImplementedError()
        else:
            with amp.autocast(enabled=args["half"]):
                with torch.no_grad():
                    metrics = eval_model(net, filepath)
            with sequence_metrics_path.open("wb") as f:
                f.write(json.dumps(metrics, indent=2).encode())
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
    parent_parser.add_argument("-n", "--dry-run", action="store_true", help="dry run")
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
    subparsers = parser.add_subparsers(help="model source", dest="source")
    subparsers.required = True

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
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
        dest="path",
        type=str,
        required=True,
        help="checkpoint path",
    )
    return parser


def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)

    if args.source == "pretrained":
        model = load_pretrained(args.architecture, args.metric, args.qualities[0])
        # runs = sorted(args.qualities)
        # opts = (args.architecture, args.metric)
        # load_func = load_pretrained
        # log_fmt = "\rEvaluating {0} | {run:d}"
    elif args.source == "checkpoint":
        model = load_checkpoint(args.architecture, args.path)
        # runs = args.paths
        # opts = (args.architecture,)
        # load_func = load_checkpoint
        # log_fmt = "\rEvaluating {run:s}"

    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")
        if args.half:
            model = model.half()

    args = vars(args)
    results = run_inference(args.pop("dataset"), model, args.pop("output"), **args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
