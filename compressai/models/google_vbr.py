import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models.google import ScaleHyperprior, MeanScaleHyperprior, JointAutoregressiveHierarchicalPriors
from compressai.entropy_models import EntropyBottleneck, EntropyBottleneckVbr
from compressai.ops import quantize_ste, LowerBound
from .utils import update_registered_buffers
from .base import get_scale_table
from compressai.registry import register_model
eps = 1e-9


@register_model("bmshj2018-hyperprior-vbr")
class ScaleHyperpriorVbr(ScaleHyperprior):
    r""" Vraiable bitrate (vbr) version ala DCC24
    """

    def __init__(self, N,  M, vr_entbttlnck=False, **kwargs):
        
        super().__init__(N=N, M=M, **kwargs)

        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18]
        gg = 1.0 * torch.tensor([0.10000, 0.13944, 0.19293, 0.26874, 0.37268, 0.51801, 0.71957, 1.00000])
        self.Gain = torch.nn.Parameter(gg, requires_grad=True)  # inverse of quantization step size
        self.levels = len(self.lmbda)  
        # 3/4-layer NN to get quant offset from Gain and std i.e. scales_hat, a single NN
        Nds = 12 
        self.QuantABCD = nn.Sequential(nn.Linear(1+1, Nds), nn.ReLU(), 
                                       nn.Linear(Nds, Nds), nn.ReLU(), 
                                       nn.Linear(Nds,   1), )
        # flag to indicate whether to use or not to use (ie. old codepath of qvrf) quantization offsets
        self.no_quantoffset = True
        # use also variable rate hyper prior z (entropy_bottleneck)
        self.vr_entbttlnck = vr_entbttlnck
        if self.vr_entbttlnck:
            self.entropy_bottleneck = EntropyBottleneckVbr(N) # EntropyBottleneckVariableQuantization(N)
            Ndsz = 10
            self.gayn2zqstep = nn.Sequential(nn.Linear(   1, Ndsz), nn.ReLU(), 
                                             nn.Linear(Ndsz, Ndsz), nn.ReLU(), 
                                             nn.Linear(Ndsz,    1), nn.Softplus())
            self.lower_bound_zqstep = LowerBound(0.5)  


    def forward(self, x, noise=False, stage=3, s=1, inputscale=0):
        # fatih: disables this 
        # gg = 1.0 * torch.tensor([0.10000, 0.13944, 0.19293, 0.26874, 0.37268, 0.51801, 0.71957, 1.00000])
        # self.Gain = torch.nn.Parameter(gg, requires_grad=False)
        # fatih: modifying the logic here
        # if stage > 1:
        #     if s != 0:
        #         scale = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps
        #     else:
        #         s = 0
        #         scale = self.Gain[s].detach()
        #         # scale = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps # fatih: must use this, if above with detach is used, error occurs in backw prop with model weights frozen
        # else:
        #     scale = self.Gain[0].detach()
        # New logic below
        s = max(0, min(s, len(self.Gain)-1)) # clip s to correct range
        if self.training:
            if stage > 1:
                if s != (len(self.Gain) - 1):
                    scale = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps
                else:
                    s = len(self.Gain) - 1
                    # scale = self.Gain[s].detach() # fatih: fix this gain to 1 
                    scale = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps
            else:
                s = len(self.Gain) - 1
                scale = self.Gain[s].detach()
        else:
            if inputscale == 0:
                scale = self.Gain[s].detach()
            else:
                scale = inputscale
        
        rescale = 1.0 / scale.clone().detach() # fatih: should we use detach() here or not ?

        if noise: # fatih: NOTE, modifications to support q_offsets in noise case are not well tested
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.h_s(z_hat)
            if self.no_quantoffset:
                # fatih: the two lines below (quant to get y_hat, entropy to get y_likelihoods) are the original codes, I will modify them with gaussian_conditional_variable
                y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize") * rescale
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
                # y_hat, y_likelihoods = self.gaussian_conditional.forward_variable(y, scales_hat, means=None, qs=scale) # no need to rescale (during training and validation ONLY), done in quantization
            else:
                # fatih: below is the code that performs quantization with quant offset. Two lines above are the original code without quant offset.
                y_zm_sc = ((y - 0) * scale).detach()
                signs = torch.sign(y_zm_sc)
                q_abs = quantize_ste(torch.abs(y_zm_sc))
                q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

                stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)

                q_offsets[q_abs < 0.0001] = 0   # must use zero offset for locations quantized to zero !
                ###q_offsets[(y-0).detach().mean(dim=(2,3)).abs() > 2.0] = 0  # also use zero offset for channels with large mean since the zero mean assumption does not hold for these channels 
                
                q_offsets *= signs  # since we add offsets directly to non-abs of y_tilde

                y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize") * rescale + q_offsets * rescale
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
                # fatih: above is the code that performs quantization with quant offset
            x_hat = self.g_s(y_hat)
        else:  # STE
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            # fatih: variable rate entropy bottleneck ?
            if not self.vr_entbttlnck:
                z_hat, z_likelihoods = self.entropy_bottleneck(z)

                z_offset = self.entropy_bottleneck._get_medians()
                z_tmp = z - z_offset
                z_hat = quantize_ste(z_tmp) + z_offset
            else:
                z_qstep = self.gayn2zqstep(1.0 / scale.clone().view(1))
                z_qstep = self.lower_bound_zqstep(z_qstep)
                z_hat, z_likelihoods = self.entropy_bottleneck(z, qs=z_qstep[0], training=None, ste=False) # ste=True) # this is EBVariableQuantization class, and should take care of ste quantization

            scales_hat = self.h_s(z_hat)
            if self.no_quantoffset:
                # fatih: the two lines below are the original codes, I will modify them with gaussian_conditional_variable
                y_hat = quantize_ste(y * scale, "ste") * rescale
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
                # y_hat, y_likelihoods = self.gaussian_conditional.forward_variable(y, scales_hat, means=None, qs=scale, qmode="ste")  # !!! STE 
            else:
                # fatih: below is the code that performs quantization with quant offset. Two lines above are the original code without quant offset.
                y_ch_means =  0  # y.mean(dim=(0,2,3), keepdim=True)
                y_zm = y - y_ch_means 
                y_zm_sc = y_zm * scale 
                signs = torch.sign(y_zm_sc).detach()
                q_abs = quantize_ste(torch.abs(y_zm_sc))
                q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

                stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)

                q_offsets[q_abs < 0.0001] = 0   # must use zero offset for locations quantized to zero !
                
                y_hat = signs * (q_abs + q_offsets)
                y_hat = y_hat * rescale + y_ch_means
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
                # fatih: above is the code that performs quantization with quant offset
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict, vr_entbttlnck=False):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M, vr_entbttlnck)  # fatih: add vr_bottleneck here
        if 'QuantOffset' in state_dict.keys():
            del state_dict["QuantOffset"]
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False, scale=None):  # fatih: support also scale input for variable EntropyBottleneck 
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        # updated |= super().update(force=force, sc=scale)
        if isinstance(self.entropy_bottleneck, EntropyBottleneckVbr): # fatih: modeified to support also Variable Quantization
            sc = scale
            if sc is None:
                rv = self.entropy_bottleneck.update(force=force)
            else:
                z_qstep = self.gayn2zqstep(1.0 / sc.view(1))
                z_qstep = self.lower_bound_zqstep(z_qstep)
                rv = self.entropy_bottleneck.update_variable(force=force, qs=z_qstep)
        elif isinstance(self.entropy_bottleneck, EntropyBottleneck):
            rv = self.entropy_bottleneck.update(force=force)
        updated |= rv
        return updated

    def compress(self, x, s, inputscale=0):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        # fatih: disables this 
        # gg = 1.0 * torch.tensor([0.10000, 0.13944, 0.19293, 0.26874, 0.37268, 0.51801, 0.71957, 1.00000])
        # self.Gain = torch.nn.Parameter(gg, requires_grad=False)
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        if not self.vr_entbttlnck: # fatih: support for variable rate EntropyBottleneck
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        else:
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_strings = self.entropy_bottleneck.compress(z, qs=z_qstep[0])
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], qs=z_qstep[0])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
        y_strings = self.gaussian_conditional.compress(y * scale, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s, inputscale):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale

        if not self.vr_entbttlnck: # fatih: support for variable rate EntropyBottleneck
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        else:
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape, qs=z_qstep[0])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
        if self.no_quantoffset:
            # fatih: the one line below is the original code. I will modify them to do quantization with quant offset
            y_hat = self.gaussian_conditional.decompress(strings[0], indexes) * rescale
        else:
            # fatih: below is the code that performs quantization with quant offset. The one line above is the original code without quant offset.
            q_val = self.gaussian_conditional.decompress(strings[0], indexes)
            q_abs, signs = q_val.abs(), torch.sign(q_val)
            
            # q_offsets = torch.broadcast_to(self.QuantOffset[s, :].view(1, -1, 1, 1), q_abs.shape).clone()
            q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

            stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
            q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)

            q_offsets[q_abs< 0.0001] = 0  # must use zero offset for locations quantized to zero !
            ###? q_offsets[y_zm.mean(dim=(2,3)).abs() > 2.0] = 0  #  ?????? perhaps send the indexes to zero out in an input argument simply ? their transmission bits shouldbe ignorable.

            y_hat = signs * (q_abs + q_offsets)
            y_ch_means =  0 
            y_hat = y_hat * rescale + y_ch_means
            # fatih: above is the code that performs quantization with quant offset.
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}