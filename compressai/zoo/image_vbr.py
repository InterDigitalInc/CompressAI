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

from torch.hub import load_state_dict_from_url

from compressai.models import (
    JointAutoregressiveHierarchicalPriorsVbr,
    MeanScaleHyperpriorVbr,
    ScaleHyperpriorVbr,
)

from .pretrained import load_pretrained

__all__ = [
    "bmshj2018_hyperprior_vbr",
    "mbt2018_mean_vbr",
    "mbt2018_vbr",
]

model_architectures = {
    "bmshj2018-hyperprior-vbr": ScaleHyperpriorVbr,
    "mbt2018-mean-vbr": MeanScaleHyperpriorVbr,
    "mbt2018-vbr": JointAutoregressiveHierarchicalPriorsVbr,
}

root_url = "https://compressai.s3.amazonaws.com/models/v1"
model_urls = {
    "bmshj2018-hyperprior-vbr": {
        "mse": {1: f"{root_url}/bmshj2018-hyperprior-blabla.pth.tar"},
        "ms-ssim": {1: f"{root_url}/bmshj2018-hyperprior-blabla.pth.tar"},
    },
    "mbt2018-mean-vbr": {
        "mse": {1: f"{root_url}/bmshj2018-hyperprior-blabla.pth.tar"},
        "ms-ssim": {1: f"{root_url}/bmshj2018-hyperprior-blabla.pth.tar"},
    },
    "mbt2018-vbr": {
        "mse": {1: f"{root_url}/bmshj2018-hyperprior-blabla.pth.tar"},
        "ms-ssim": {1: f"{root_url}/bmshj2018-hyperprior-blabla.pth.tar"},
    },
}

cfgs = {
    "bmshj2018-hyperprior-vbr": {
        1: (192, 320),
    },
    "mbt2018-mean-vbr": {
        1: (192, 320),
    },
    "mbt2018-vbr": {
        1: (192, 320),
    },
}


def _load_model(
    architecture, metric, quality, pretrained=False, progress=True, **kwargs
):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if (
            architecture not in model_urls
            or metric not in model_urls[architecture]
            or quality not in model_urls[architecture][metric]
        ):
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric][quality]
        state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = load_pretrained(state_dict)
        model = model_architectures[architecture].from_state_dict(state_dict)
        return model

    model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)
    return model


def bmshj2018_hyperprior_vbr(
    quality, metric="mse", pretrained=False, progress=True, **kwargs
):
    r"""Bla bla...

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', no 'ms-ssim' yet)
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse"):  # ("mse", "ms-ssim"): # we have only mse model
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be 1')

    return _load_model(
        "bmshj2018-hyperprior-vbr", metric, quality, pretrained, progress, **kwargs
    )


def mbt2018_mean_vbr(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Bla bla...

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', no 'ms-ssim' yet)
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse"):  # ("mse", "ms-ssim"): # we have only mse model
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be 1')

    return _load_model(
        "mbt2018-mean-vbr", metric, quality, pretrained, progress, **kwargs
    )


def mbt2018_vbr(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Bla bla...

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', no 'ms-ssim' yet)
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse"):  # ("mse", "ms-ssim"): # we have only mse model
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be 1')

    return _load_model("mbt2018-vbr", metric, quality, pretrained, progress, **kwargs)
