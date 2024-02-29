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

import math

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import amp

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import QReLU
from compressai.ops import quantize_ste
from compressai.registry import register_model

from ..base import CompressionModel
from ..utils import conv, deconv, gaussian_blur, gaussian_kernel2d, meshgrid2d


@register_model("ssf2020")
class ScaleSpaceFlow(CompressionModel):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class HyperEncoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                )

        class HyperDecoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class HyperDecoderWithQReLU(nn.Module):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__()

                def qrelu(input, bit_depth=8, beta=100):
                    return QReLU.apply(input, bit_depth, beta)

                self.deconv1 = deconv(in_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu1 = qrelu
                self.deconv2 = deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu2 = qrelu
                self.deconv3 = deconv(mid_planes, out_planes, kernel_size=5, stride=2)
                self.qrelu3 = qrelu

            def forward(self, x):
                x = self.qrelu1(self.deconv1(x))
                x = self.qrelu2(self.deconv2(x))
                x = self.qrelu3(self.deconv3(x))

                return x

        class Hyperprior(CompressionModel):
            def __init__(self, planes: int = 192, mid_planes: int = 192):
                super().__init__()
                self.entropy_bottleneck = EntropyBottleneck(mid_planes)
                self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
                self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
                self.hyper_decoder_scale = HyperDecoderWithQReLU(
                    planes, mid_planes, planes
                )
                self.gaussian_conditional = GaussianConditional(None)

            def forward(self, y):
                z = self.hyper_encoder(y)
                z_hat, z_likelihoods = self.entropy_bottleneck(z)

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                _, y_likelihoods = self.gaussian_conditional(y, scales, means)
                y_hat = quantize_ste(y - means) + means
                return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

            def compress(self, y):
                z = self.hyper_encoder(y)

                z_string = self.entropy_bottleneck.compress(z)
                z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)

                indexes = self.gaussian_conditional.build_indexes(scales)
                y_string = self.gaussian_conditional.compress(y, indexes, means)
                y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)

                return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

            def decompress(self, strings, shape):
                assert isinstance(strings, list) and len(strings) == 2
                z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_hat = self.gaussian_conditional.decompress(
                    strings[0], indexes, z_hat.dtype, means
                )

                return y_hat

        self.img_encoder = Encoder(3)
        self.img_decoder = Decoder(3)
        self.img_hyperprior = Hyperprior()

        self.res_encoder = Encoder(3)
        self.res_decoder = Decoder(3, in_planes=384)
        self.res_hyperprior = Hyperprior()

        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1)
        self.motion_hyperprior = Hyperprior()

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []
        frames_likelihoods = []

        x_hat, likelihoods = self.forward_keyframe(frames[0])
        reconstructions.append(x_hat)
        frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, likelihoods = self.forward_inter(x, x_ref)
            reconstructions.append(x_ref)
            frames_likelihoods.append(likelihoods)

        return {
            "x_hat": reconstructions,
            "likelihoods": frames_likelihoods,
        }

    def forward_keyframe(self, x):
        y = self.img_encoder(x)
        y_hat, likelihoods = self.img_hyperprior(y)
        x_hat = self.img_decoder(y_hat)
        return x_hat, {"keyframe": likelihoods}

    def encode_keyframe(self, x):
        y = self.img_encoder(x)
        y_hat, out_keyframe = self.img_hyperprior.compress(y)
        x_hat = self.img_decoder(y_hat)

        return x_hat, out_keyframe

    def decode_keyframe(self, strings, shape):
        y_hat = self.img_hyperprior.decompress(strings, shape)
        x_hat = self.img_decoder(y_hat)

        return x_hat

    def forward_inter(self, x_cur, x_ref):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)
        y_motion_hat, motion_likelihoods = self.motion_hyperprior(y_motion)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        y_res_hat, res_likelihoods = self.res_hyperprior(y_res)

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        return x_rec, {"motion": motion_likelihoods, "residual": res_likelihoods}

    def encode_inter(self, x_cur, x_ref):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)
        y_motion_hat, out_motion = self.motion_hyperprior.compress(y_motion)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        y_res_hat, out_res = self.res_hyperprior.compress(y_res)

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        return x_rec, {
            "strings": {
                "motion": out_motion["strings"],
                "residual": out_res["strings"],
            },
            "shape": {"motion": out_motion["shape"], "residual": out_res["shape"]},
        }

    def decode_inter(self, x_ref, strings, shapes):
        key = "motion"
        y_motion_hat = self.motion_hyperprior.decompress(strings[key], shapes[key])

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        key = "residual"
        y_res_hat = self.res_hyperprior.decompress(strings[key], shapes[key])

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    @amp.autocast(enabled=False)
    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

    def compress(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        frame_strings = []
        shape_infos = []

        x_ref, out_keyframe = self.encode_keyframe(frames[0])

        frame_strings.append(out_keyframe["strings"])
        shape_infos.append(out_keyframe["shape"])

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref)

            frame_strings.append(out_interframe["strings"])
            shape_infos.append(out_interframe["shape"])

        return frame_strings, shape_infos

    def decompress(self, strings, shapes):
        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f"Invalid number of frames: {len(strings)}.")

        assert len(strings) == len(
            shapes
        ), f"Number of information should match {len(strings)} != {len(shapes)}."

        dec_frames = []

        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)

        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape)
            dec_frames.append(x_ref)

        return dec_frames

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net
