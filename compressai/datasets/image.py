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

import enum
import os
import random
import re

from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from compressai.transforms.functional import ycbcr2rgb, yuv_420_to_444


class VideoFormat(enum.Enum):
    YUV400 = "yuv400"  # planar 4:0:0 YUV
    YUV420 = "yuv420"  # planar 4:2:0 YUV
    YUV422 = "yuv422"  # planar 4:2:2 YUV
    YUV444 = "yuv444"  # planar 4:4:4 YUV
    RGB = "rgb"  # planar 4:4:4 RGB


# Table of "fourcc" formats from Vooya, GStreamer, and ffmpeg mapped to a normalized enum value.
video_formats = {
    "yuv400": VideoFormat.YUV400,
    "yuv420": VideoFormat.YUV420,
    "p420": VideoFormat.YUV420,
    "i420": VideoFormat.YUV420,
    "yuv422": VideoFormat.YUV422,
    "p422": VideoFormat.YUV422,
    "i422": VideoFormat.YUV422,
    "y42B": VideoFormat.YUV422,
    "yuv444": VideoFormat.YUV444,
    "p444": VideoFormat.YUV444,
    "y444": VideoFormat.YUV444,
}


framerate_to_fraction = {
    "23.98": Fraction(24000, 1001),
    "23.976": Fraction(24000, 1001),
    "29.97": Fraction(30000, 1001),
    "59.94": Fraction(60000, 1001),
}

file_extensions = {
    "yuv",
    "rgb",
    "raw",
}


subsampling = {
    VideoFormat.YUV400: (0, 0),
    VideoFormat.YUV420: (2, 2),
    VideoFormat.YUV422: (2, 1),
    VideoFormat.YUV444: (1, 1),
}


bitdepth_to_dtype = {
    8: np.uint8,
    10: np.uint16,
    12: np.uint16,
    14: np.uint16,
    16: np.uint16,
}


def get_raw_video_file_info(filename: str) -> Dict[str, Any]:
    """
    Deduce size, framerate, bitdepth, and format from the filename based on the
    Vooya specifcation.

    This is defined as follows:

        youNameIt_WIDTHxHEIGHT[_FPS[Hz|fps]][_BITSbit][_(P420|P422|P444|UYVY|YUY2|YUYV|I444)].[rgb|yuv|bw|rgba|bgr|bgra … ]

    See: <https://www.offminor.de/vooya-usage.html#vf>

    Additional support for the GStreamer and ffmpeg format string deduction is
    also supported (I420_10LE and yuv420p10le for example).
    See: <https://gstreamer.freedesktop.org/documentation/video/video-format.html?gi-language=c#GstVideoFormat>

    Returns (dict):
        Dictionary containing width, height, framerate, bitdepth, and format
        information if found.
    """
    size_pattern = r"(?P<width>\d+)x(?P<height>\d+)"
    framerate_pattern = r"(?P<framerate>[\d\.]+)(?:Hz|fps)"
    bitdepth_pattern = r"(?P<bitdepth>\d+)bit"
    formats = "|".join(video_formats.keys())
    format_pattern = (
        rf"(?P<format>{formats})(?:[p_]?(?P<bitdepth2>\d+)(?P<endianness>LE|BE))?"
    )
    extension_pattern = rf"(?P<extension>{'|'.join(file_extensions)})"
    cut_pattern = "([0-9]+)-([0-9]+)"
    pattern = re.compile(
        rf"(?:_{size_pattern})?(?:_{framerate_pattern})?(?:_{bitdepth_pattern})?(?:_{format_pattern})?(?:_{cut_pattern})?\.{extension_pattern}$",
        flags=re.IGNORECASE,
    )
    match = pattern.search(filename)
    if match:
        info: Dict[str, str] = match.groupdict()
    else:
        return {}

    if info["bitdepth"] and info["bitdepth2"] and info["bitdepth"] != info["bitdepth2"]:
        raise ValueError(f'Filename "{filename}" specifies bit-depth twice.')

    if info["bitdepth2"]:
        info["bitdepth"] = info["bitdepth2"]
    del info["bitdepth2"]

    outinfo: Dict[str, Union[str, int, float, Fraction, VideoFormat]] = {}
    outinfo.update(info)

    # Normalize the format
    if info["format"] is not None:
        outinfo["format"] = video_formats.get(info["format"].lower(), info["format"])

    if info["endianness"] is not None:
        outinfo["endianness"] = info["endianness"].lower()

    if info["framerate"] is not None:
        framerate = info["framerate"]
        if framerate in framerate_to_fraction:
            outinfo["framerate"] = framerate_to_fraction[framerate]
        else:
            outinfo["framerate"] = Fraction(framerate)

    for key in ("width", "height", "bitdepth"):
        if info.get(key) is not None:
            outinfo[key] = int(info[key])

    return outinfo


def readRawYUVFrame(fp, skip_frms, frmSize, dtype, format):
    width, height = frmSize

    w_sub, h_sub = subsampling[format]
    if h_sub > 1:
        sub_height = (height + 1) // h_sub
    elif h_sub:
        sub_height = round(height / h_sub)
    else:
        sub_height = 0

    if w_sub > 1:
        sub_width = (width + 1) // w_sub if w_sub else 0
    elif w_sub:
        sub_width = round(width / w_sub)
    else:
        sub_width = 0

    y_size = width * height
    c_size = sub_width * sub_height

    dsize_per_frame = y_size + 2 * c_size

    fp.seek((skip_frms * dsize_per_frame), 0)

    raw = np.frombuffer(fp.read(y_size), dtype=dtype)
    y_plane = raw.reshape((height, width))

    raw = np.frombuffer(fp.read(c_size), dtype=dtype)
    u_plane = raw.reshape((sub_height, sub_width))

    raw = np.frombuffer(fp.read(c_size), dtype=dtype)
    v_plane = raw.reshape((sub_height, sub_width))

    return [y_plane, u_plane, v_plane]


def get_num_frms(file, width, height, video_format, dtype):
    w_sub, h_sub = subsampling[video_format]
    itemsize = np.array([0], dtype=dtype).itemsize

    frame_size = (width * height) + 2 * (
        round(width / w_sub) * round(height / h_sub)
    ) * itemsize

    file_size = os.path.getsize(file)

    total_num_frms = file_size // frame_size

    return total_num_frms


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class VideoFolder(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directorie containing mnay sub-directorie like Vimeo90K Dataset:

    .. code-block::

        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    ...
                    -0932/
                    -0933/
                    ...
                - 00011/
                    ...
                - 00012/
                    ...

    training and testing (valid) clips are withdrew from sub-directory navigated by
    corresponding input files listing relevant folders.

    This class returns a set of three video frames in a tuple.
    Random interval can be applied to if subfolders includes more than 6 frames.

    Args:
        root (string): root directory of the dataset
        rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        split="train",
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")

        splitfile = Path(f"{root}/{split}.list")
        splitdir = Path(f"{root}/sequences")

        if not splitfile.is_file():
            raise RuntimeError(f'Invalid file "{root}"')

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        with open(splitfile, "r") as f_in:
            self.sample_folders = [Path(f"{splitdir}/{f.strip()}") for f in f_in]

        self.max_frames = 3  # hard coding for now
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]

        frames = [self.transform(Image.open(p)) for p in frame_paths]

        if self.rnd_temp_order:
            if random.random() < 0.5:
                return frames[::-1]

        return frames

    def __len__(self):
        return len(self.sample_folders)


class VideoSequenceInYUV420(Dataset):
    """Load frames from a video sequence to encode a clip

    This class returns a frame in RGB channels.
    Interval can be applied while drawing a frame.

    Args:
        interval (bool): interval when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        src_addr : a file address for input video
        video_resolution : video resolution in a tuple (width, height)
        bitdepth : bitdepth per pixel for input video data
    """

    def __init__(
        self,
        interval=1,
        src_addr=None,
        resolution=None,
        bitdepth=8,
        video_format=None,
        num_frms=-1,
    ):

        height, width = resolution

        self.fp = None
        try:
            self.fp = open(src_addr, "rb")
        except IOError:
            raise RuntimeError(f"File does not exist at {src_addr}")

        info = get_raw_video_file_info(src_addr)

        self.bitdepth = bitdepth if bitdepth else info.get("bitdepth", None)
        self.format = video_format if video_format else info.get("format", None)
        self.height = height if height else info.get("height", None)
        self.width = width if width else info.get("width", None)
        self.max_value = 2 ** self.bitdepth - 1

        if width is None or height is None or bitdepth is None or format is None:
            raise RuntimeError(f"Could not get sequence information from {src_addr}")

        self.dtype = bitdepth_to_dtype[bitdepth]
        self.interval = interval

        if num_frms == -1:
            num_frms = get_num_frms(
                src_addr, self.width, self.height, self.format, self.dtype
            )

        self.num_frms = num_frms

    def close(self):
        if self.fp:
            self.fp.close()

    @staticmethod
    def _to_tensor(x, max_value: int):
        out = torch.from_numpy(np.true_divide(x, max_value, dtype="float32"))
        # HW -> CHW (C=1)
        out = out.unsqueeze(0)
        return out

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        frm_idx = index * self.interval
        frame = readRawYUVFrame(
            self.fp, frm_idx, (self.width, self.height), self.dtype, self.format
        )

        yuv_planes = [self._to_tensor(c, self.max_value).unsqueeze(0) for c in frame]
        rgb_img = ycbcr2rgb(yuv_420_to_444(yuv_planes, mode="bilinear"))
        return rgb_img

        return self.transform(frame)

    def __len__(self):
        return self.num_frms
