"""State-dict introspection helpers for SWAtten-based context heads.

Used by MambaIC / MambaVC ``from_state_dict`` to recover ``window_size``,
``head_dim`` and ``inter_dim`` (a.k.a. ``support_attention_dim``) from a
checkpoint. Each model used to ship a private ``_infer_*`` copy of these.
"""

from __future__ import annotations

import math

from typing import Dict

from torch import Tensor

__all__ = [
    "infer_swatten_attention_dim",
    "infer_swatten_head_dim",
    "infer_swatten_window_size",
]


def infer_swatten_window_size(
    state_dict: Dict[str, Tensor], prefix: str, *, default: int = 8
) -> int:
    for key, tensor in state_dict.items():
        if key.startswith(prefix) and key.endswith("relative_position_bias_table"):
            return (math.isqrt(tensor.size(0)) + 1) // 2
    return default


def infer_swatten_head_dim(
    state_dict: Dict[str, Tensor],
    prefix: str,
    hidden_channels: int,
    *,
    default: int = 16,
) -> int:
    for key, tensor in state_dict.items():
        if key.startswith(prefix) and key.endswith("relative_position_bias_table"):
            return hidden_channels // tensor.size(1)
    return default


def infer_swatten_attention_dim(
    state_dict: Dict[str, Tensor],
    prefix: str,
    *,
    default: int = 128,
) -> int:
    key = f"{prefix}.in_conv.weight"
    if key in state_dict:
        return state_dict[key].size(0)
    return default
