# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/JiangWeibeta/MLIC
# (originally distributed under the Apache License 2.0). Modifications by
# InterDigital Communications, Inc. are released under the BSD 3-Clause Clear
# License terms below.

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

from .context import (
    ChannelContext,
    LinearGlobalInterContext,
    LinearGlobalIntraContext,
    LocalContext,
    StackedCheckerboardConv,
    VanillaGlobalInterContext,
    VanillaGlobalIntraContext,
    WindowCheckerboardAttn,
)
from .transforms import (
    AnalysisTransform,
    EntropyParameters,
    HyperAnalysis,
    HyperSynthesis,
    LatentResidualPrediction,
    SynthesisTransform,
)
from .utils import (
    checkerboard_anchor,
    checkerboard_merge,
    checkerboard_nonanchor,
    checkerboard_split,
    compress_symbols,
    decompress_symbols,
    squeeze_anchor,
    squeeze_nonanchor,
    unsqueeze_anchor,
    unsqueeze_nonanchor,
)

__all__ = [
    "AnalysisTransform",
    "ChannelContext",
    "EntropyParameters",
    "HyperAnalysis",
    "HyperSynthesis",
    "LatentResidualPrediction",
    "LinearGlobalInterContext",
    "LinearGlobalIntraContext",
    "LocalContext",
    "SynthesisTransform",
    "StackedCheckerboardConv",
    "VanillaGlobalInterContext",
    "VanillaGlobalIntraContext",
    "WindowCheckerboardAttn",
    "checkerboard_anchor",
    "checkerboard_merge",
    "checkerboard_nonanchor",
    "checkerboard_split",
    "compress_symbols",
    "decompress_symbols",
    "squeeze_anchor",
    "squeeze_nonanchor",
    "unsqueeze_anchor",
    "unsqueeze_nonanchor",
]
