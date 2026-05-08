from .inference import (
    infer_swatten_attention_dim,
    infer_swatten_head_dim,
    infer_swatten_window_size,
)
from .swin import (
    WMSA,
    ConvTransBlock,
    PatchMerging,
    PatchSplit,
    SWAtten,
    SwinBlock,
    WindowAttention,
    WinNoShiftAttention,
    WinResidualUnit,
    build_window_attention_mask,
    pad_to_window_multiple,
    window_partition,
    window_reverse,
)

__all__ = [
    "ConvTransBlock",
    "PatchMerging",
    "PatchSplit",
    "SWAtten",
    "SwinBlock",
    "WMSA",
    "WinNoShiftAttention",
    "WinResidualUnit",
    "WindowAttention",
    "build_window_attention_mask",
    "infer_swatten_attention_dim",
    "infer_swatten_head_dim",
    "infer_swatten_window_size",
    "pad_to_window_multiple",
    "window_partition",
    "window_reverse",
]


def __getattr__(name):
    if name == "Win_noShift_Attention":
        from .swin import Win_noShift_Attention as _alias

        return _alias
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
