import argparse
import json
import subprocess
import sys
import tempfile

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from pytorch_msssim import ms_ssim  # type: ignore
from torch import Tensor
from torch.utils.model_zoo import tqdm

from compressai.transforms.functional import ycbcr2rgb, yuv_420_to_444

from .codecs import H264, H265
from .rawvideo import RawVideoSequence, VideoFormat

codec_classes = [H264, H265]


Frame = Tuple[Tensor, Tensor, Tensor]


def to_tensors(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray], max_value: int = 1
) -> Frame:
    return (
        torch.from_numpy(np.true_divide(frame[0], max_value, dtype="float32")),
        torch.from_numpy(np.true_divide(frame[1], max_value, dtype="float32")),
        torch.from_numpy(np.true_divide(frame[2], max_value, dtype="float32")),
    )


# def sequence_to_tensor(sequence):
#     max_value = 2 ** sequence.bitdepth - 1
#     return tuple(frame_to_tensor(frame, max_value) for frame in sequence)


# class SequenceToTensor:
#     def __call__(self, sequence):
#         return sequence_to_tensor(sequence)


def run_cmdline(
    cmdline: List[Any], logpath: Optional[Path] = None, dry_run: bool = False
) -> None:
    cmdline = list(map(str, cmdline))
    print(f"--> Running: {' '.join(cmdline)}", file=sys.stderr)

    if dry_run:
        return

    if logpath is None:
        out = subprocess.check_output(cmdline).decode()
        if out:
            print(out)
        return

    p = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with logpath.open("w") as f:
        if p.stdout is not None:
            for bline in p.stdout:
                line = bline.decode()
                sys.stdout.write(line)
                f.write(line)
    p.wait()


def compute_metrics_for_frame(
    org_frame: Frame,
    dec_frame: Frame,
    max_val: float = 1.0,
) -> Dict[str, Any]:
    org_frame = tuple(p.unsqueeze(0).unsqueeze(0) for p in org_frame)  # type: ignore
    dec_frame = tuple(p.unsqueeze(0).unsqueeze(0) for p in dec_frame)  # type:ignore
    out = {}

    # YCbCr metrics
    for i, component in enumerate("yuv"):
        mse = (org_frame[i] - dec_frame[i]).pow(2).mean().item()
        out[f"{component}_mse"] = mse

    # RGB metrics
    # org = ycbcr2rgb(yuv_420_to_444(org_frame, mode="bicubic"))  # type: ignore
    # dec = ycbcr2rgb(yuv_420_to_444(dec_frame, mode="bicubic"))  # type: ignore

    # org = (org * max_val).clamp(0, max_val).round()
    # dec = (dec * max_val).clamp(0, max_val).round()

    # mse = (org - dec).pow(2).mean().item()
    # ms_ssim_val = ms_ssim(org, dec, data_range=max_val).item()
    mse = 100.0
    ms_ssim_val = 100.0
    out.update({"ms-ssim": ms_ssim_val, "mse": mse})
    return out


def get_filesize(filepath: Union[Path, str]) -> int:
    return Path(filepath).stat().st_size


def evaluate(
    org_seq_path: Path, dec_seq_path: Path, bitstream_path: Path
) -> Dict[str, Any]:
    # load original and decoded sequences
    org_seq = RawVideoSequence.from_file(str(org_seq_path))
    dec_seq = RawVideoSequence.new_like(org_seq, str(dec_seq_path))

    max_val = 2 ** org_seq.bitdepth - 1
    num_frames = len(org_seq)

    if len(dec_seq) != num_frames:
        raise RuntimeError(
            "Invalid number of frames in decoded sequence "
            f"({num_frames}!={len(dec_seq)})"
        )

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    # compute metrics for each frame
    results = defaultdict(list)
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            org_frame = to_tensors(org_seq[i])
            dec_frame = to_tensors(dec_seq[i])
            metrics = compute_metrics_for_frame(org_frame, dec_frame, max_val)
            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    # compute average metrics for sequence
    seq_results: Dict[str, Any] = {k: np.mean(v) for k, v in results.items()}
    filesize = get_filesize(bitstream_path)
    seq_results["bpp"] = 8.0 * filesize / (num_frames * org_seq.width * org_seq.height)
    seq_results["rgb_psnr"] = 20 * np.log10(max_val) - 10 * np.log10(seq_results["mse"])
    for component in "yuv":
        seq_results[f"{component}_psnr"] = 20 * np.log10(max_val) - 10 * np.log10(
            seq_results[f"{component}_mse"]
        )
    return seq_results


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


def create_parser() -> Tuple[
    argparse.ArgumentParser, argparse.ArgumentParser, argparse._SubParsersAction
]:
    parser = argparse.ArgumentParser(
        description="Video codec baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="sequences directory")
    parent_parser.add_argument("output", type=str, help="output directory")
    parent_parser.add_argument("-n", "--dry-run", action="store_true", help="dry run")
    parent_parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite previous runs"
    )
    subparsers = parser.add_subparsers(dest="codec", help="video codec")
    subparsers.required = True
    return parser, parent_parser, subparsers


def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser, parent_parser, subparsers = create_parser()

    codec_lookup = {}
    for cls in codec_classes:
        codec = cls()
        codec_lookup[codec.name] = codec
        codec_parser = subparsers.add_parser(
            codec.name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[parent_parser],
        )
        codec.add_parser_args(codec_parser)

    args, extra = parser.parse_known_args(args)

    # create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    codec = codec_lookup[args.codec]
    results_paths = []
    for filepath in sorted(Path(args.dataset).glob("*.yuv")):
        encode_cmd = codec.get_encode_cmd(filepath, args)
        outputpath = codec.get_outputpath(filepath, args)

        # encode sequence if not already encoded
        if args.force:
            outputpath.unlink(missing_ok=True)
        if not outputpath.is_file():
            logpath = outputpath.with_suffix(".log")
            run_cmdline(encode_cmd, logpath=logpath, dry_run=args.dry_run)

        # compute metrics if not already performed
        sequence_metrics_path = outputpath.with_suffix(".json")
        results_paths.append(sequence_metrics_path)

        if args.force:
            sequence_metrics_path.unlink(missing_ok=True)
        if sequence_metrics_path.is_file():
            continue

        with tempfile.NamedTemporaryFile(suffix=".yuv", delete=True) as f:
            # decode sequence
            cmd = ["ffmpeg", "-y", "-i", outputpath, "-pix_fmt", "yuv420p", f.name]
            run_cmdline(cmd)

            # compute metrics
            metrics = evaluate(filepath, Path(f.name), outputpath)
            with sequence_metrics_path.open("wb") as f:
                f.write(json.dumps(metrics, indent=2).encode())

    results = aggregate_results(results_paths)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
