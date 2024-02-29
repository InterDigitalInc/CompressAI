# Copyright (c) 2021-2024, InterDigital Communications, Inc
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

import enum
import re

from fractions import Fraction
from typing import Any, Dict, Sequence, Union

import numpy as np


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
    "420": VideoFormat.YUV420,
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


def make_dtype(format, value_type, width, height):
    # Use float division with rounding to account for oddly sized Y planes
    # and even sized U and V planes to match ffmpeg.

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

    return np.dtype(
        [
            ("y", value_type, (height, width)),
            ("u", value_type, (sub_height, sub_width)),
            ("v", value_type, (sub_height, sub_width)),
        ]
    )


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

    patterns = (
        size_pattern,
        framerate_pattern,
        bitdepth_pattern,
        format_pattern,
        cut_pattern,
        extension_pattern,
    )
    info: Dict[str, Any] = {}
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            info.update(match.groupdict())

    if not info:
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


def get_num_frms(file_size, width, height, video_format, dtype):
    w_sub, h_sub = subsampling[video_format]
    itemsize = np.array([0], dtype=dtype).itemsize

    frame_size = (width * height) + 2 * (
        round(width / w_sub) * round(height / h_sub)
    ) * itemsize

    total_num_frms = file_size // frame_size

    return total_num_frms


class RawVideoSequence(Sequence[np.ndarray]):
    """
    Generalized encapsulation of raw video buffer data that can hold RGB or
    YCbCr with sub-sampling.

    Args:
        data: Single dimension array of the raw video data.
        width: Video width, if not given it may be deduced from the filename.
        height: Video height, if not given it may be deduced from the filename.
        bitdepth: Video bitdepth, if not given it may be deduced from the filename.
        format: Video format, if not given it may be deduced from the filename.
        framerate: Video framerate, if not given it may be deduced from the filename.
    """

    def __init__(
        self,
        mmap: np.memmap,
        width: int,
        height: int,
        bitdepth: int,
        format: VideoFormat,
        framerate: int,
    ):
        self.width = width
        self.height = height
        self.bitdepth = bitdepth
        self.framerate = framerate

        if isinstance(format, str):
            self.format = video_formats[format.lower()]
        else:
            self.format = format

        value_type = bitdepth_to_dtype[bitdepth]
        self.dtype = make_dtype(
            self.format, value_type=value_type, width=width, height=height
        )
        self.data = mmap.view(self.dtype)

        self.total_frms = get_num_frms(mmap.size, width, height, format, value_type)

    @classmethod
    def new_like(
        cls, sequence: "RawVideoSequence", filename: str
    ) -> "RawVideoSequence":
        mmap = np.memmap(filename, dtype=bitdepth_to_dtype[sequence.bitdepth], mode="r")
        return cls(
            mmap,
            width=sequence.width,
            height=sequence.height,
            bitdepth=sequence.bitdepth,
            format=sequence.format,
            framerate=sequence.framerate,
        )

    @classmethod
    def from_file(
        cls,
        filename: str,
        width: int = None,
        height: int = None,
        bitdepth: int = None,
        format: VideoFormat = None,
        framerate: int = None,
    ) -> "RawVideoSequence":
        """
        Loads a raw video file from the given filename.

        Args:
            filename: Name of file to load.
            width: Video width, if not given it may be deduced from the filename.
            height: Video height, if not given it may be deduced from the filename.
            bitdepth: Video bitdepth, if not given it may be deduced from the filename.
            format: Video format, if not given it may be deduced from the filename.

        Returns (RawVideoSequence):
            A RawVideoSequence instance wrapping the file on disk with a
            np memmap.
        """
        info = get_raw_video_file_info(filename)

        bitdepth = bitdepth if bitdepth else info.get("bitdepth", None)
        format = format if format else info.get("format", None)
        height = height if height else info.get("height", None)
        width = width if width else info.get("width", None)
        framerate = framerate if framerate else info.get("framerate", None)

        if width is None or height is None or bitdepth is None or format is None:
            raise RuntimeError(f"Could not get sequence information {filename}")

        mmap = np.memmap(filename, dtype=bitdepth_to_dtype[bitdepth], mode="r")

        return cls(
            mmap,
            width=width,
            height=height,
            bitdepth=bitdepth,
            format=format,
            framerate=framerate,
        )

    def __getitem__(self, index: Union[int, slice]) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def close(self):
        del self.data
