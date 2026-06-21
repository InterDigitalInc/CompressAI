"""Convert an MLIC-family checkpoint to compressai layout.

Loads an MLIC-family checkpoint, instantiates the matching compressai model
through ``from_state_dict``, and optionally writes the converted state dict.
MLIC++ checkpoints from JiangWeibeta/MLIC are translated from the published
root-level layout to the containerized ``latent_codec.*`` layout inside
``convert_upstream_mlicpp_state_dict``.

The upstream-checkpoint conversion helpers live in this example CLI (not in
``compressai.models.mlic``) so the model module stays a clean compressai-native
definition. ``examples/`` is not an importable package, so tests load
``convert_upstream_mlicpp_state_dict`` by file path.

Example::

    python examples/convert_mlic_checkpoint.py \
        --src candidate/MLIC/mlicpp_mse_q5_2960000.pth.tar \
        --variant mlicpp \
        --dst /tmp/mlicpp_compressai.pth \
        --smoke
"""

from __future__ import annotations

import argparse
import re

from pathlib import Path
from typing import Dict, Iterable, List, Type

import torch
import torch.nn as nn

from torch import Tensor

from compressai.models.mlic import (
    MLIC,
    MLICPlus,
    MLICPlusPlus,
    MLICv2,
)

_VARIANTS: Dict[str, Type[nn.Module]] = {
    "mlic": MLIC,
    "mlic+": MLICPlus,
    "mlicpp": MLICPlusPlus,
    "mlicv2": MLICv2,
}


# ---------------------------------------------------------------------------
# Upstream MLIC++ checkpoint conversion
#
# The old fork-script layout stored the hyperprior and per-slice modules under
# a monolithic ``latent_codec``. The compressai model follows the ELIC-style
# container structure: ``HyperpriorLatentCodec`` owns ``h_a`` / ``h_s`` / ``z``,
# while ``latent_codec.y`` owns the channel groups and per-slice checkerboard
# leaves.
# ---------------------------------------------------------------------------

_CURRENT_SLICE_RE = re.compile(r"^latent_codec\.y\.latent_codec\.y(\d+)\.")
_LEGACY_SLICE_RE = re.compile(
    r"^(?:latent_codec\.)?"
    r"(?:local_context|channel_context|global_inter_context|"
    r"global_intra_context|entropy_parameters_anchor|"
    r"entropy_parameters_nonanchor|lrp_anchor|lrp_nonanchor)\.(\d+)\."
)

_ROOT_TO_CONTAINER_PREFIXES: Dict[str, str] = {
    "h_a.": "latent_codec.h_a.",
    "h_s.": "latent_codec.h_s.",
    "entropy_bottleneck.": "latent_codec.z.entropy_bottleneck.",
}

_LEGACY_LIST_RENAMES: Dict[str, str] = {
    "local_context": "latent_codec.y.latent_codec.y{index}.spatial_context_nonanchor.",
    "channel_context": "latent_codec.y.channel_context.y{index}.channel_part.",
    "global_inter_context": "latent_codec.y.channel_context.y{index}.global_inter_part.",
    "global_intra_context": (
        "latent_codec.y.latent_codec.y{index}.intra_channel_context_nonanchor."
    ),
    "entropy_parameters_anchor": (
        "latent_codec.y.latent_codec.y{index}.entropy_parameters_anchor."
    ),
    "entropy_parameters_nonanchor": (
        "latent_codec.y.latent_codec.y{index}.entropy_parameters_nonanchor."
    ),
    "lrp_anchor": "latent_codec.y.latent_codec.y{index}.lrp_anchor.",
    "lrp_nonanchor": "latent_codec.y.latent_codec.y{index}.lrp_nonanchor.",
}


def _strip_data_parallel_prefix(key: str) -> str:
    if key.startswith("module."):
        return key[len("module.") :]
    return key


def _infer_slice_num(keys: Iterable[str]) -> int:
    indices: List[int] = []
    for raw_key in keys:
        key = _strip_data_parallel_prefix(raw_key)
        for pattern in (_CURRENT_SLICE_RE, _LEGACY_SLICE_RE):
            match = pattern.match(key)
            if match is not None:
                indices.append(int(match.group(1)))
                break
    return max(indices) + 1 if indices else 10


def _convert_mlicpp_key(
    key: str,
    *,
    slice_num: int,
) -> List[str]:
    key = _strip_data_parallel_prefix(key)

    for old, new in _ROOT_TO_CONTAINER_PREFIXES.items():
        if key.startswith(old):
            return [new + key[len(old) :]]

    if key.startswith("latent_codec.entropy_bottleneck."):
        return [
            "latent_codec.z.entropy_bottleneck."
            + key[len("latent_codec.entropy_bottleneck.") :]
        ]

    for prefix in ("gaussian_conditional.", "latent_codec.gaussian_conditional."):
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            return [
                f"latent_codec.y.latent_codec.y{index}.y.gaussian_conditional." + suffix
                for index in range(slice_num)
            ]

    legacy_key = key
    if legacy_key.startswith("latent_codec."):
        legacy_key = legacy_key[len("latent_codec.") :]
    parts = legacy_key.split(".", 2)
    if len(parts) == 3 and parts[0] in _LEGACY_LIST_RENAMES:
        name, index, suffix = parts
        return [_LEGACY_LIST_RENAMES[name].format(index=index) + suffix]

    return [key]


def convert_upstream_mlicpp_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Convert legacy MLIC++ checkpoint keys to the containerized layout.

    The old fork-script layout stored the hyperprior and per-slice modules
    under a monolithic ``latent_codec``. The compressai model follows the
    ELIC-style container structure: ``HyperpriorLatentCodec`` owns ``h_a`` /
    ``h_s`` / ``z``, while ``latent_codec.y`` owns the channel groups and
    per-slice checkerboard leaves.
    """
    slice_num = _infer_slice_num(state_dict.keys())
    converted: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        for new_key in _convert_mlicpp_key(key, slice_num=slice_num):
            converted[new_key] = value
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream or already-converted MLIC-family checkpoint.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(_VARIANTS),
        default="mlicpp",
        help="Model variant to instantiate (default: mlicpp).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help=(
            "Optional output path for the converted state dict. If omitted, "
            "the script only verifies that the checkpoint loads cleanly."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a forward smoke test on a synthetic image.",
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=256,
        help="Synthetic square image size used by --smoke (default: 256).",
    )
    return parser.parse_args()


def load_state_dict(path: Path) -> Dict[str, Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise SystemExit(f"checkpoint does not contain a state dict: {path}")
    return dict(state_dict)


def make_synthetic_image(size: int) -> Tensor:
    if size <= 0:
        raise SystemExit("--smoke-size must be positive")
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, size),
        torch.linspace(0, 1, size),
        indexing="ij",
    )
    return (
        torch.stack(
            [
                0.5 + 0.3 * torch.sin(8 * xs),
                0.5 + 0.3 * torch.sin(8 * ys),
                0.5 + 0.3 * torch.cos(8 * (xs + ys)),
            ],
            dim=0,
        )
        .unsqueeze(0)
        .clamp(0, 1)
    )


def run_smoke(net: nn.Module, size: int) -> None:
    img = make_synthetic_image(size)
    with torch.no_grad():
        out = net(img)
    n_pix = size * size
    psnr = -10 * torch.log10(((out["x_hat"].clamp(0, 1) - img) ** 2).mean()).item()
    y_bpp = -torch.log2(out["likelihoods"]["y"]).sum().item() / n_pix
    z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
    print(
        f"smoke: PSNR={psnr:.2f}dB  y_bpp={y_bpp:.4f}  z_bpp={z_bpp:.4f}  "
        f"total_bpp={y_bpp + z_bpp:.4f}"
    )


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"checkpoint not found: {args.src}")

    state_dict = load_state_dict(args.src)
    if args.variant == "mlicpp":
        state_dict = convert_upstream_mlicpp_state_dict(state_dict)
    print(f"loaded checkpoint -> {len(state_dict)} compressai keys")

    cls = _VARIANTS[args.variant]
    net = cls.from_state_dict(state_dict).eval()
    print(
        "variant: "
        f"{args.variant}, N={net.N}, M={net.M}, slice_num={net.slice_num}, "
        f"context_window={getattr(net, 'context_window', None)}, "
        f"local_kernel={getattr(net, 'local_kernel', None)}, "
        f"local_layers={getattr(net, 'local_layers', None)}"
    )
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dst)
        print(f"wrote converted state dict -> {args.dst}")

    if args.smoke:
        run_smoke(net, args.smoke_size)


if __name__ == "__main__":
    main()
