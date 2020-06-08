import torch
import torch.nn as nn

from .bound_ops import LowerBound


class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """
    def __init__(self, minimum=0, reparam_offset=2**-18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset**2
        self.register_buffer('pedestal', torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2)**.5
        self.lower_bound = LowerBound(bound)

    def init(self, x):
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x):
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out
