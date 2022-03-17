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

import abc
import argparse
import platform
import subprocess
import sys

from pathlib import Path
from typing import Any, List

from compressai.datasets.rawvideo import RawVideoSequence, get_raw_video_file_info


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
    # name = ""
    description = ""
    help = ""

    @classmethod
    def setup_args(cls, parser):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    @property
    def description(self):
        return self._description

    def add_parser_args(self, parser: argparse.ArgumentParser) -> None:
        pass

    def set_args(self, args):
        return args

    @abc.abstractmethod
    def get_bin_path(self, filepath: Path, **args: Any) -> Path:
        raise NotImplementedError

    @abc.abstractmethod
    def get_encode_cmd(self, filepath: Path, **args: Any) -> List[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_decode_cmd(self, filepath: Path, **args: Any) -> List[Any]:
        raise NotImplementedError


def get_ffmpeg_version():
    rv = run_command(["ffmpeg", "-version"])
    return rv.split()[2]


class x264(Codec):
    preset = ""
    tune = ""

    @property
    def name(self):
        return "x264"

    def description(self):
        return f"{self.name} {self.preset}, {self.tune}, ffmpeg version {get_ffmpeg_version()}"

    def name_config(self):
        return f"{self.name}-{self.preset}-tune-{self.tune}"

    def add_parser_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("-p", "--preset", default="medium", help="preset")
        parser.add_argument(
            "--tune",
            default="psnr",
            help="tune encoder for psnr or ssim (default: %(default)s)",
        )

    def set_args(self, args):
        args = super().set_args(args)
        self.preset = args.preset
        self.tune = args.tune
        return args

    def get_bin_path(self, filepath: Path, qp, binpath: str) -> Path:
        return Path(binpath) / (
            f"{filepath.stem}_{self.name}_{self.preset}_tune-{self.tune}_qp{qp}.mp4"
        )

    def get_encode_cmd(self, filepath: Path, qp, bindir) -> List[Any]:
        info = get_raw_video_file_info(filepath.stem)
        binpath = self.get_bin_path(filepath, qp, bindir)
        cmd = [
            "ffmpeg",
            "-y",
            "-s:v",
            f"{info['width']}x{info['height']}",
            "-i",
            filepath,
            "-c:v",
            "h264",
            "-crf",
            qp,
            "-preset",
            self.preset,
            "-bf",
            0,
            "-tune",
            self.tune,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "4",
            binpath,
        ]
        return cmd

    def get_decode_cmd(
        self, binpath: Path, decpath: Path, input_filepath: Path
    ) -> List[Any]:
        del input_filepath  # unused here
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            binpath,
            "-pix_fmt",
            "yuv420p",
            decpath,
        ]
        return cmd


class x265(x264):
    @property
    def name(self):
        return "x265"

    def get_encode_cmd(self, filepath: Path, qp, bindir) -> List[Any]:
        info = get_raw_video_file_info(filepath.stem)
        binpath = self.get_bin_path(filepath, qp, bindir)
        cmd = [
            "ffmpeg",
            "-s:v",
            f"{info['width']}x{info['height']}",
            "-i",
            filepath,
            "-c:v",
            "hevc",
            "-crf",
            qp,
            "-preset",
            self.preset,
            "-x265-params",
            "bframes=0",
            "-tune",
            self.tune,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "4",
            binpath,
        ]
        return cmd


class VTM(Codec):
    """VTM: VVC reference software"""

    binext = "bin"
    config = ""

    @property
    def name(self):
        return "VTM"

    def description(self):
        return f"VTM reference software, version {self.get_version(self.encoder_path)}"

    def name_config(self):
        return f"{self.name}-v{self.get_version(self.encoder_path)}-{self.config}"

    def get_version(selfm, encoder_path):
        rv = run_command([encoder_path, "--help"], ignore_returncodes=[1])
        version = rv.split(b"\n")[1].split()[4].decode().strip("[]")
        return version

    def get_encoder_path(self, build_dir):
        system = platform.system()
        try:
            elfnames = {"Darwin": "EncoderApp", "Linux": "EncoderAppStatic"}
            return Path(build_dir) / elfnames[system]
        except KeyError as err:
            raise RuntimeError(f'Unsupported platform "{system}"') from err

    def get_decoder_path(self, build_dir):
        system = platform.system()
        try:
            elfnames = {"Darwin": "DecoderApp", "Linux": "DecoderAppStatic"}
            return Path(build_dir) / elfnames[system]
        except KeyError as err:
            raise RuntimeError(f'Unsupported platform "{system}"') from err

    @classmethod
    def add_parser_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-b",
            "--build-dir",
            type=str,
            required=True,
            help="VTM build dir",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            required=True,
            help="VTM config file",
        )
        parser.add_argument(
            "--rgb", action="store_true", help="Use RGB color space (over YCbCr)"
        )

    def set_args(self, args):
        args = super().set_args(args)
        self.encoder_path = self.get_encoder_path(args.build_dir)
        self.decoder_path = self.get_decoder_path(args.build_dir)
        self.config_path = args.config
        self.config = Path(self.config_path).stem.split("_")[1]
        self.version = self.get_version(self.encoder_path)
        self.rgb = args.rgb
        return args

    def get_encode_cmd(self, filepath: Path, qp, bindir) -> List[Any]:
        info = get_raw_video_file_info(filepath.stem)
        num_frames = len(RawVideoSequence.from_file(str(filepath)))
        binpath = self.get_bin_path(filepath, qp, bindir)
        cmd = [
            self.encoder_path,
            "-i",
            filepath,
            "-c",
            self.config_path,
            "-q",
            qp,
            "-o",
            "/dev/null",
            "-b",
            binpath,
            "-wdt",
            info["width"],
            "-hgt",
            info["height"],
            "-fr",
            info["framerate"],
            "-f",
            num_frames,
            f'--InputBitDepth={info["bitdepth"]}',
            f'--OutputBitDepth={info["bitdepth"]}',
            # "--ConformanceWindowMode=1",
        ]

        if self.rgb:
            cmd += [
                "--InputColourSpaceConvert=RGBtoGBR",
                "--SNRInternalColourSpace=1",
                "--OutputInternalColourSpace=0",
            ]
        return cmd

    def get_bin_path(self, filepath: Path, qp, binpath: str) -> Path:
        return Path(binpath) / (
            f"{filepath.stem}_{self.name}_{self.config}_qp{qp}.{self.binext}"
        )

    def get_decode_cmd(
        self, binpath: Path, decpath: Path, input_filepath: Path
    ) -> List[Any]:
        output_bitdepth = get_raw_video_file_info(input_filepath.stem)["bitdepth"]
        cmd = [self.decoder_path, "-b", binpath, "-o", decpath, "-d", output_bitdepth]
        return cmd


class HM(VTM):
    """HM: HEVC reference software"""

    binext = "bin"
    config = ""

    @property
    def name(self):
        return "HM"

    def description(self):
        return f"HM reference software, version {self.get_version(self.encoder_path)}"

    def name_config(self):
        return f"{self.name}-v{self.get_version(self.encoder_path)}-{self.config}"

    def get_encoder_path(self, build_dir):
        system = platform.system()
        try:
            elfnames = {"Darwin": "TAppEncoder", "Linux": "TAppEncoderStatic"}
            return Path(build_dir) / elfnames[system]
        except KeyError as err:
            raise RuntimeError(f'Unsupported platform "{system}"') from err

    def get_decoder_path(self, build_dir):
        system = platform.system()
        try:
            elfnames = {"Darwin": "TAppDecoder", "Linux": "TAppDecoderStatic"}
            return Path(build_dir) / elfnames[system]
        except KeyError as err:
            raise RuntimeError(f'Unsupported platform "{system}"') from err

    def set_args(self, args):
        args = super().set_args(args)
        self.encoder_path = self.get_encoder_path(args.build_dir)
        self.decoder_path = self.get_decoder_path(args.build_dir)
        self.config_path = args.config
        self.config = Path(self.config_path).stem.split("_")[1]
        self.version = self.get_version(self.encoder_path)
        self.rgb = args.rgb
        return args

    def get_encode_cmd(self, filepath: Path, qp, bindir) -> List[Any]:
        info = get_raw_video_file_info(filepath.stem)
        num_frames = len(RawVideoSequence.from_file(str(filepath)))
        binpath = self.get_bin_path(filepath, qp, bindir)
        cmd = [
            self.encoder_path,
            "-i",
            filepath,
            "-c",
            self.config_path,
            "-q",
            qp,
            "-o",
            "/dev/null",
            "-b",
            binpath,
            "-wdt",
            info["width"],
            "-hgt",
            info["height"],
            "-fr",
            info["framerate"],
            "-f",
            num_frames,
            f'--InputBitDepth={info["bitdepth"]}',
            f'--OutputBitDepth={info["bitdepth"]}',
            # "--ConformanceWindowMode=1",
        ]

        if self.rgb:
            cmd += [
                "--InputColourSpaceConvert=RGBtoGBR",
                "--SNRInternalColourSpace=1",
                "--OutputInternalColourSpace=0",
            ]
        return cmd

    def get_decode_cmd(
        self, binpath: Path, decpath: Path, input_filepath: Path
    ) -> List[Any]:
        output_bitdepth = get_raw_video_file_info(input_filepath.stem)["bitdepth"]
        cmd = [self.decoder_path, "-b", binpath, "-o", decpath, "-d", output_bitdepth]
        return cmd
