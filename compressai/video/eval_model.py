import argparse
import json
import math
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lvc.models.networks.video.ssf import google2020_ssf  # type: ignore

# from pytorch_msssim import ms_ssim  # type: ignore
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm

from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)

from .eval_codec import aggregate_results, convert_legal_to_full_range, to_tensors
from .rawvideo import RawVideoSequence, VideoFormat

models = {"google2020_ssf": google2020_ssf}

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]


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


def eval_model(net: nn.Module, sequence: Path) -> Dict[str, Any]:
    org_seq = RawVideoSequence.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(net.parameters()).device
    num_frames = len(org_seq)
    num_pixels = org_seq.height * org_seq.width
    results = defaultdict(list)

    # output = Path(sequence.name).open("wb")
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            cur_frame = convert_frame(org_seq[i], device, org_seq.bitdepth)
            cur_frame, padding = pad(cur_frame)

            if i == 0:
                rec_frame, likelihoods = net.forward_keyframe(cur_frame)  # type:ignore
            else:
                rec_frame, likelihoods = net.forward_inter(
                    cur_frame, rec_frame
                )  # type:ignore
            rec_frame = rec_frame.clamp(0, 1)

            metrics = compute_metrics_for_frame(
                crop(cur_frame, padding), crop(rec_frame, padding), org_seq.bitdepth
            )
            metrics["bpp"] = sum(
                (torch.log(getattr(lkl, k)).sum() / (-math.log(2) * num_pixels))
                for lkl in likelihoods.values()
                for k in ("y", "z")
            )

            # out = yuv_444_to_420(rgb2ycbcr(crop(rec_frame, padding).clamp(0, 1)))
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
    # pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    # pretrained_parser.add_argument(
    #     "-m",
    #     "--metric",
    #     type=str,
    #     choices=["mse", "ms-ssim"],
    #     default="mse",
    #     help="metric trained against (default: %(default)s)",
    # )
    # pretrained_parser.add_argument(
    #     "-q",
    #     "--quality",
    #     dest="qualities",
    #     nargs="+",
    #     type=int,
    #     default=(1,),
    # )

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

    if args.source != "checkpoint":
        raise NotImplementedError()

    model = load_checkpoint(args.architecture, args.path)
    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")
        if args.half:
            model = model.half()

    args = vars(args)
    results = run_inference(args.pop("dataset"), model, args.pop("output"), **args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
