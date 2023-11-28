import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models.google import ScaleHyperprior, MeanScaleHyperprior, JointAutoregressiveHierarchicalPriors
from compressai.entropy_models import EntropyBottleneck, EntropyBottleneckVbr
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.ops import quantize_ste, LowerBound
from .utils import update_registered_buffers, conv, deconv
from .base import get_scale_table
from compressai.registry import register_model
eps = 1e-9


@register_model("bmshj2018-hyperprior-vbr")
class ScaleHyperpriorVbr(ScaleHyperprior):
    r""" Variable bitrate (vbr) version a la DCC24
    """

    def __init__(self, N,  M, vr_entbttlnck=False, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        # lambdas to use during training
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18]
        self.levels = len(self.lmbda)  
        # gain: inverse of quantization step size
        self.Gain = torch.nn.Parameter(torch.tensor(
            [0.10000, 0.13944, 0.19293, 0.26874, 0.37268, 0.51801, 0.71957, 1.00000]), 
            requires_grad=True)
        # 3-layer NN to get quant offset from Gain and stdev i.e. scales_hat
        Nds = 12 
        self.QuantABCD = nn.Sequential(nn.Linear(1+1, Nds), nn.ReLU(), 
                                       nn.Linear(Nds, Nds), nn.ReLU(), 
                                       nn.Linear(Nds,   1), )
        # flag to indicate whether to use or not to use quantization offsets
        self.no_quantoffset = False
        # use also variable rate hyper prior z (entropy_bottleneck)
        self.vr_entbttlnck = vr_entbttlnck
        if self.vr_entbttlnck:
            self.entropy_bottleneck = EntropyBottleneckVbr(N)
            # 3-layer NN to get quant step size for hyper prior from Gain 
            Ndsz = 10
            self.gayn2zqstep = nn.Sequential(nn.Linear(   1, Ndsz), nn.ReLU(), 
                                             nn.Linear(Ndsz, Ndsz), nn.ReLU(), 
                                             nn.Linear(Ndsz,    1), nn.Softplus())
            self.lower_bound_zqstep = LowerBound(0.5)  

    def forward(self, x, noise=False, stage=3, s=1, inputscale=0):
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

    def update(self, scale_table=None, force=False, scale=None):
        # fatih: add also scale input for variable-rate (vr) EntropyBottleneck 
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        # fatih: update vr EntropyBottleneck with given scale, i.e. quantization step size
        if isinstance(self.entropy_bottleneck, EntropyBottleneckVbr): 
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
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        if not self.vr_entbttlnck: 
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        else:  # fatih: support vr EntropyBottleneck
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

        if not self.vr_entbttlnck: 
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        else:  # fatih: support vr EntropyBottleneck
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape, qs=z_qstep[0])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
        if self.no_quantoffset:
            y_hat = self.gaussian_conditional.decompress(strings[0], indexes) * rescale
        else:
            q_val = self.gaussian_conditional.decompress(strings[0], indexes)
            q_abs, signs = q_val.abs(), torch.sign(q_val)
            
            q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

            stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
            q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)

            q_offsets[q_abs< 0.0001] = 0  # must use zero offset for locations quantized to zero

            y_hat = signs * (q_abs + q_offsets)
            y_ch_means =  0 
            y_hat = y_hat * rescale + y_ch_means
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    

@register_model("mbt2018-mean-vbr")
# class MeanScaleHyperpriorVbr(ScaleHyperpriorVbr):
class MeanScaleHyperpriorVbr(ScaleHyperpriorVbr, MeanScaleHyperprior):
    r""" Variable bitrate (vbr) version a la DCC24
    """

    def __init__(self, N=192,  M=320, vr_entbttlnck=False, **kwargs):
        super().__init__(N, M, vr_entbttlnck=vr_entbttlnck, **kwargs)

    def forward(self, x, noise=False, stage=3, s=1, inputscale=0):
        # fatih: New logic below
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
        
        rescale = 1.0 / scale.clone().detach()  # fatih: should we use detach() here or not ?

        if noise: # fatih: NOTE, modifications to support q_offsets in noise case are not well tested
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            if self.no_quantoffset:
                # fatih: the two lines below (quant to get y_hat, entropy to get y_likelihoods) are the original codes, I will modify them with gaussian_conditional_variable
                y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize", means=means_hat * scale) * rescale + means_hat
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale, means=means_hat * scale)
                # y_hat, y_likelihoods = self.gaussian_conditional.forward_variable(y, scales_hat, means=None, qs=scale) # no need to rescale (during training and validation ONLY), done in quantization
            else:
                # fatih: below is the code that performs quantization with quant offset. Two lines above are the original code without quant offset.
                y_zm_sc = ((y - means_hat) * scale).detach()
                signs = torch.sign(y_zm_sc)
                q_abs = quantize_ste(torch.abs(y_zm_sc))
                q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

                stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)

                q_offsets[q_abs < 0.0001] = 0   # must use zero offset for locations quantized to zero !
                
                q_offsets *= signs  # since we add offsets directly to non-abs of y_tilde

                y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize", means=means_hat * scale) * rescale + means_hat + q_offsets * rescale
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale, means=means_hat * scale)
                # fatih: above is the code that performs quantization with quant offset
            x_hat = self.g_s(y_hat)
        else:  # STE
            y = self.g_a(x)
            z = self.h_a(y)
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

            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            if self.no_quantoffset:
                # fatih: the two lines below are the original codes, I will modify them with gaussian_conditional_variable
                y_hat = self.quantizer.quantize((y - means_hat) * scale, "ste") * rescale + means_hat 
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale, means=means_hat * scale)
                # y_hat, y_likelihoods = self.gaussian_conditional.forward_variable(y, scales_hat, means=None, qs=scale, qmode="ste")  # !!! STE 
            else:
                # fatih: below is the code that performs quantization with quant offset. Two lines above are the original code without quant offset.
                y_zm = y - means_hat
                y_zm_sc = y_zm * scale 
                signs = torch.sign(y_zm_sc).detach()
                q_abs = quantize_ste(torch.abs(y_zm_sc))
                q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

                stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)
               
                q_offsets[q_abs < 0.0001] = 0   # must use zero offset for locations quantized to zero !
        
                y_hat = signs * (q_abs + q_offsets)
                y_hat = y_hat * rescale + means_hat
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale, means=means_hat * scale)
                # fatih: above is the code that performs quantization with quant offset
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, s, inputscale=0):
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale
        y = self.g_a(x)
        z = self.h_a(y)

        if not self.vr_entbttlnck: 
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        else:  # fatih: support for variable rate EntropyBottleneck
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_strings = self.entropy_bottleneck.compress(z, qs=z_qstep[0])
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], qs=z_qstep[0])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
        y_strings = self.gaussian_conditional.compress(y * scale, indexes, means=means_hat * scale)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s, inputscale):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale

        if not self.vr_entbttlnck: 
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        else: # fatih: support for variable rate EntropyBottleneck
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape, qs=z_qstep[0])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
        if self.no_quantoffset:
            y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat * scale) * rescale
        else:
            q_val = self.gaussian_conditional.decompress(strings[0], indexes)
            q_abs, signs = q_val.abs(), torch.sign(q_val)

            q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

            stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
            q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)

            q_offsets[q_abs< 0.0001] = 0  # must use zero offset for locations quantized to zero 

            y_hat = signs * (q_abs + q_offsets)
            y_hat = y_hat * rescale + means_hat 
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



@register_model("mbt2018-vbr")
class JointAutoregressiveHierarchicalPriorsVbr(ScaleHyperpriorVbr, JointAutoregressiveHierarchicalPriors):
    r""" Variable bitrate (vbr) version a la DCC24
    """

    def __init__(self, N=192,  M=320, **kwargs):
        super().__init__(N, M, vr_entbttlnck=False, **kwargs)

        self.ste_recursive = True
        # flag to indicate whether this will be used or not
        self.scl2ctx = True 
        # this generates from scale (i.e. quant step) a tensor that will be added 
        # to context_prediction output so that it changes depending on scale in order to 
        # have better entropy parameters for any quantization stepsize
        self.scale_to_context = nn.Linear(1, 2 * M)

    
    def forward(self, x, noise=False, stage=3, s=1, inputscale=0):
        # # fatih: modifying the logic here
        # if stage > 1:
        #     if s != 0:
        #         scale = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps
        #     else:
        #         s = 0
        #         scale = self.Gain[s].detach()
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

        rescale = 1.0 / scale.clone().detach()

        if noise: # fatih: NOTE, modifications to support q_offsets in noise case are not well tested
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)
            if self.no_quantoffset:
                y_hat = self.gaussian_conditional.quantize(y*scale, "noise" if self.training else "dequantize") * rescale
                ctx_params = self.context_prediction(y_hat)
                gaussian_params = self.entropy_parameters(
                    torch.cat((params, ctx_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                _, y_likelihoods = self.gaussian_conditional(y*scale - means_hat*scale, scales_hat*scale)
            else:
                y_hat = self.gaussian_conditional.quantize(y*scale, "noise" if self.training else "dequantize") * rescale
                ctx_params = self.context_prediction(y_hat)
                gaussian_params = self.entropy_parameters(
                    torch.cat((params, ctx_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                y_zm_sc = ((y - means_hat) * scale)
                signs = torch.sign(y_zm_sc).detach()
                q_abs = quantize_ste(torch.abs(y_zm_sc))
                q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

                stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)

                q_offsets[q_abs < 0.0001] = 0   # must use zero offset for locations quantized to zero !
                
                q_offsets *= signs  # since we add offsets directly to non-abs of y_tilde

                y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize", means=means_hat * scale) * rescale + means_hat + q_offsets * rescale
                _, y_likelihoods = self.gaussian_conditional(y * scale - means_hat * scale, scales_hat * scale)

            x_hat = self.g_s(y_hat)
        else:  # STE
            y = self.g_a(x)
            z = self.h_a(y)
            _, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

            params = self.h_s(z_hat)
            # ste_recursive = False  # fatih: try to implement a non-recursive training/validation code with STE
            if self.ste_recursive:
                kernel_size = 5  # context prediction kernel size
                padding = (kernel_size - 1) // 2
                y_hat = F.pad(y, (padding, padding, padding, padding))
                # fatih: all modifications to perform quantization with quant offset are inside the _stequantization()
                y_hat, y_likelihoods = self._stequantization(y_hat, params, y.size(2), y.size(3), kernel_size, padding, scale, rescale)
                # y_hat, y_likelihoods = self._stequantization_v2(y_hat, params, y.size(2), y.size(3), kernel_size, padding, scale, rescale)
            else:
                y_hat = self.gaussian_conditional.quantize((y*scale), "noise" if self.training else "dequantize") * rescale  # dont do .detach() on y*scale
                ctx_params = self.context_prediction(y_hat)
                gaussian_params = self.entropy_parameters(
                    torch.cat((params, ctx_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale, means=means_hat * scale) # fatih: NOTE: likelihoods are found from reconstructions without quant_offset !

                if self.no_quantoffset:
                    y_hat = quantize_ste((y - means_hat) * scale) * rescale + means_hat
                else:
                    y_zm = y - means_hat  # means_hat  # means_hat.detach()  # fatih: do not detach() here, o.w. no gradient flow from mse-cost to means_hat etc
                    y_zm_sc = y_zm * scale 
                    signs = torch.sign(y_zm_sc).detach()
                    q_abs = quantize_ste(torch.abs(y_zm_sc))

                    # q_offsets = torch.broadcast_to(self.QuantOffset[s, :].view(1, -1, 1, 1), q_abs.shape).clone()
                    q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

                    stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                    q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)
                    q_offsets[q_abs < 0.0001] = 0   # must use zero offset for locations quantized to zero !

                    y_hat = signs * (q_abs + q_offsets)
                    y_hat = y_hat * rescale + means_hat 

            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def _stequantization(self, y_hat, params, height, width, kernel_size, padding, scale, rescale):
        y_likelihoods = torch.zeros([y_hat.size(0), y_hat.size(1), height, width]).to(scale.device)
        # fatih: get ctx_scl to condition entropy model also on scale
        if self.scl2ctx:
            ctx_scl = self.scale_to_context.forward(scale.view(1,1)).view(1,-1,1,1)
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size].clone()
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )
                # fatih: adjust ctx_p to be conditioned on also scale so that entropy params are also conditioned on scale ... 
                if self.scl2ctx:
                    p = params[:, :, h: h + 1, w: w + 1]
                    gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p + ctx_scl), dim=1))
                else:
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[:, :, h: h + 1, w: w + 1]
                    gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))

                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_crop = y_crop[:, :, padding, padding]
                _, y_likelihoods[:, :, h: h + 1, w: w + 1] = self.gaussian_conditional(
                    ((y_crop - means_hat) * scale).unsqueeze(2).unsqueeze(3),
                    (scales_hat * scale).unsqueeze(2).unsqueeze(3))
                if self.no_quantoffset:
                    y_q = self.quantizer.quantize((y_crop - means_hat.detach()) * scale, 
                                                  "ste") * rescale + means_hat.detach()
                else:
                    y_zm = y_crop - means_hat 
                    y_zm_sc = y_zm * scale 
                    signs = torch.sign(y_zm_sc).detach()
                    q_abs = quantize_ste(torch.abs(y_zm_sc))

                    q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

                    stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=2), scale.detach().expand(q_stdev.unsqueeze(dim=2).shape)), dim=2)
                    q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=2)
                    q_offsets[q_abs < 0.0001] = 0   # must use zero offset for locations quantized to zero
            
                    y_q = signs * (q_abs + q_offsets)
                    y_q = y_q * rescale + means_hat  
                y_hat[:, :, h + padding, w + padding] = y_q
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        return y_hat, y_likelihoods
    

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, s, inputscale=0):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
                scale,
                rescale,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding, scale, rescale,):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # fatih: get ctx_scl to condition entropy model also on scale
        if self.scl2ctx:
            ctx_scl = self.scale_to_context.forward(scale.view(1,1)).view(1,-1,1,1)
        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )
                # fatih: adjust ctx_p to be conditioned on also scale so that entropy params are also conditioned on scale ...
                if self.scl2ctx:
                    ctx_p = ctx_p + ctx_scl
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)

                y_crop = y_crop[:, :, padding, padding]
                if self.no_quantoffset or (self.no_quantoffset==False and self.ste_recursive==False):
                    y_q = self.gaussian_conditional.quantize((y_crop - means_hat)* scale, "symbols")
                    y_hat[:, :, h + padding, w + padding] = (y_q) * rescale + means_hat
                else:
                    y_zm = y_crop - means_hat.detach()
                    y_zm_sc = y_zm * scale 
                    signs = torch.sign(y_zm_sc).detach()
                    q_abs = quantize_ste(torch.abs(y_zm_sc))

                    q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

                    stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=2), scale.detach().expand(q_stdev.unsqueeze(dim=2).shape)), dim=2)
                    q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=2)
                    q_offsets[q_abs < 0.0001] = 0   # must use zero offset for locations quantized to zero 

                    y_q = (signs * (q_abs + 0)).int()  
                    y_hat[:, :, h + padding, w + padding] = (signs * (q_abs + q_offsets)) * rescale + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape, s, inputscale):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
                scale,
                rescale,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
            self, y_string, y_hat, params, height, width, kernel_size, padding, scale, rescale
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        if self.no_quantoffset==False and self.ste_recursive==False:
            y_rec = torch.zeros_like(y_hat)
        # fatih: get ctx_scl to condition entropy model also on scale
        if self.scl2ctx:
            ctx_scl = self.scale_to_context.forward(scale.view(1,1)).view(1,-1,1,1)
        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # fatih: adjust ctx_p to be conditioned on also scale so that entropy params are also conditioned on scale ...
                if self.scl2ctx:
                    ctx_p = ctx_p + ctx_scl
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat* scale)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1).to(scales_hat.device)  # fatih: move rv to gpu ?
                if self.no_quantoffset:
                    rv = self.gaussian_conditional.dequantize(rv)*rescale + means_hat

                    hp = h + padding
                    wp = w + padding
                    y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv
                else:
                    q_val = self.gaussian_conditional.dequantize(rv)
                    q_abs, signs = q_val.abs(), torch.sign(q_val)

                    q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)

                    stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                    q_offsets = (-1) * (self.QuantABCD.forward( stdev_and_gain )).squeeze(dim=4)

                    q_offsets[q_abs< 0.0001] = 0  # must use zero offset for locations quantized to zero 

                    rv_out = (signs * (q_abs + q_offsets)) * rescale + means_hat

                    hp = h + padding
                    wp = w + padding
                    if self.ste_recursive==False:
                        y_hat[:, :, hp: hp + 1, wp: wp + 1] = self.gaussian_conditional.dequantize(rv)*rescale + means_hat # find also reco without quantoffset for likelihood
                        y_rec[:, :, hp: hp + 1, wp: wp + 1] = rv_out
                    else:
                        y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv_out

        # fatih: reconstruction with quantoffset will be used for only image reconstruction and reconstruction without quantoffset for likelihood
        if self.no_quantoffset==False and self.ste_recursive==False:
            y_hat = y_rec

