# Copyright (c) 2021-2025, InterDigital Communications, Inc
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

"""Causal Context Adjustment rate-distortion loss (Han et al., NeurIPS 2024).

Companion criterion for :class:`compressai.models.cca.CCAModel`. Requires
the model's ``forward`` to return ``aux_likelihoods = {"y_aux", "y_cca"}``
(populated when ``cca_training=True``) so this loss can add the auxiliary
CCA terms on top of the standard rate-distortion objective.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from pytorch_msssim import ms_ssim

from compressai.registry import register_criterion


@register_criterion("CCARateDistortionLoss")
class CCARateDistortionLoss(nn.Module):
    r"""Causal Context Adjustment rate-distortion loss from M. Han, S. Jiang,
    S. Li, X. Deng, M. Xu, C. Zhu, S. Gu: `"Causal Context Adjustment Loss
    for Learned Image Compression" <https://arxiv.org/abs/2410.04847>`_,
    Adv. in Neural Information Processing Systems 38 (NeurIPS), 2024.

    Combines the standard rate (``bpp``) and distortion (MSE / MS-SSIM)
    terms with the CCA term that measures the gap between the main and
    auxiliary causal-context likelihoods produced by
    :class:`compressai.models.cca.CCAModel` (with ``cca_training=True``).

    Args:
        lmbda: Distortion weight.
        metric: Distortion metric, ``"mse"`` or ``"ms-ssim"``.
        return_type: ``"all"`` returns the dict of components; otherwise
            return the named scalar component (e.g. ``"loss"``).
        alpha: Weight on the CCA loss term.
        beta: Weight on the bit-rate term.
    """

    def __init__(
        self,
        lmbda: float = 0.01,
        metric: str = "mse",
        return_type: str = "all",
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")

        self.lmbda = float(lmbda)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.return_type = return_type

    def forward(self, output, target):
        if "aux_likelihoods" not in output or output["aux_likelihoods"] is None:
            raise KeyError(
                "output must contain aux_likelihoods for CCARateDistortionLoss; "
                "ensure CCAModel was constructed with cca_training=True"
            )

        aux_likelihoods = output["aux_likelihoods"]
        if "y_aux" not in aux_likelihoods or "y_cca" not in aux_likelihoods:
            raise KeyError("aux_likelihoods must contain y_aux and y_cca")

        batch_size, _, height, width = target.size()
        num_pixels = batch_size * height * width
        out = {}

        out["cca_loss"] = (
            torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2))
            - torch.log(aux_likelihoods["y_cca"]).sum() / (-math.log(2))
        ) / num_pixels
        out["aux2_loss"] = torch.sum(
            aux_likelihoods["y_cca"] * torch.log(aux_likelihoods["y_aux"])
        ) / (-math.log(2) * num_pixels)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = (
            self.lmbda * distortion
            + self.beta * out["bpp_loss"]
            + self.alpha * out["cca_loss"]
            + out["aux2_loss"]
        )
        if self.return_type == "all":
            return out
        return out[self.return_type]
