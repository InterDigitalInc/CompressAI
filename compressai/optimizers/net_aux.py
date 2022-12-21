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

from typing import Any, Dict, Mapping, cast

import torch.nn as nn
import torch.optim as optim

from compressai.registry import OPTIMIZERS, register_optimizer


@register_optimizer("net_aux")
def net_aux_optimizer(
    net: nn.Module, conf: Mapping[str, Any]
) -> Dict[str, optim.Optimizer]:
    """Returns separate optimizers for net and auxiliary losses.

    Each optimizer operates on a mutually exclusive set of parameters.
    """
    parameters = {
        "net": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        },
        "aux": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        },
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters["net"] & parameters["aux"]
    union_params = parameters["net"] | parameters["aux"]
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    def make_optimizer(key):
        kwargs = dict(conf[key])
        del kwargs["type"]
        params = (params_dict[name] for name in sorted(parameters[key]))
        return OPTIMIZERS[conf[key]["type"]](params, **kwargs)

    optimizer = {key: make_optimizer(key) for key in ["net", "aux"]}

    return cast(Dict[str, optim.Optimizer], optimizer)
