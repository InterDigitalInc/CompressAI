import abc
import argparse
import os
import platform
import subprocess
import sys

from pathlib import Path
from typing import Any, List

from compressai.datasets.rawvideo import get_raw_video_file_info


def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


class Codec(abc.ABC):
    name = ""
    help = ""

    def add_parser_args(self, parser: argparse.ArgumentParser) -> None:
        pass

    @abc.abstractmethod
    def get_output_path(self, filepath: Path, **args: Any) -> Path:
        raise NotImplementedError

    @abc.abstractmethod
    def get_encode_cmd(self, filepath: Path, **args: Any) -> List[Any]:
        raise NotImplementedError


class x264(Codec):
    name = "x264"

    def add_parser_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("-p", "--preset", default="medium", help="preset")
        parser.add_argument("-q", "--qp", default=32, help="quality")

    def get_output_path(self, filepath: Path, **args: Any) -> Path:
        return Path(args["output"]) / (
            f"{filepath.stem}_{self.name}_{args['preset']}_qp{args['qp']}.mp4"
        )

    def get_encode_cmd(self, filepath: Path, **args: Any) -> List[Any]:
        info = get_raw_video_file_info(filepath.stem)
        outputpath = self.get_output_path(filepath, **args)
        cmd = [
            "ffmpeg",
            "-s:v",
            f"{info['width']}x{info['height']}",
            "-i",
            filepath,
            "-c:v",
            "h264",
            "-crf",
            args["qp"],
            "-preset",
            args["preset"],
            "-bf",
            0,
            "-tune",
            "ssim",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "4",
            outputpath,
        ]
        return cmd


class x265(x264):
    name = "x265"

    def get_encode_cmd(self, filepath: Path, **args: Any) -> List[Any]:
        info = get_raw_video_file_info(filepath.stem)
        outputpath = self.get_output_path(filepath, **args)
        cmd = [
            "ffmpeg",
            "-s:v",
            f"{info['width']}x{info['height']}",
            "-i",
            filepath,
            "-c:v",
            "hevc",
            "-crf",
            args["qp"],
            "-preset",
            args["preset"],
            "-x265-params",
            "bframes=0",
            "-tune",
            "ssim",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "4",
            outputpath,
        ]
        return cmd


def get_vtm_encoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "EncoderApp", "Linux": "EncoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err


def get_vtm_decoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "DecoderApp", "Linux": "DecoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err
