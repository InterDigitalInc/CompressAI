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

from compressai.models.video import ScaleSpaceFlow

from .pretrained import load_pretrained

__all__ = [
    "ssf2020",
]

model_architectures = {
    "ssf2020": ScaleSpaceFlow,
}

root_url = "https://compressai.s3.amazonaws.com/models/v1"
model_urls = {
    "ssf2020": {
        "mse": {
            1: f"{root_url}/ssf2020-mse-1-c1ac1a47.pth.tar",
            2: f"{root_url}/ssf2020-mse-2-79ed4e19.pth.tar",
            3: f"{root_url}/ssf2020-mse-3-9c8b998d.pth.tar",
            4: f"{root_url}/ssf2020-mse-4-577c1eda.pth.tar",
            5: f"{root_url}/ssf2020-mse-5-1dd7d574.pth.tar",
            6: f"{root_url}/ssf2020-mse-6-59dfb6f9.pth.tar",
            7: f"{root_url}/ssf2020-mse-7-4d867411.pth.tar",
            8: f"{root_url}/ssf2020-mse-8-26439e20.pth.tar",
            9: f"{root_url}/ssf2020-mse-9-e89345c4.pth.tar",
        }
    }
}


def _load_model(
    architecture, metric, quality, pretrained=False, progress=True, **kwargs
):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in range(1, 10):
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

    model = model_architectures[architecture](**kwargs)
    return model


def ssf2020(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        quality (int): Quality levels (1: lowest, highest: 9)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 9:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 9)')

    return _load_model("ssf2020", metric, quality, pretrained, progress, **kwargs)
