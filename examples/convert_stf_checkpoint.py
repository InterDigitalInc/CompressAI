"""Convert an upstream STF / WACNN checkpoint to compressai layout.

Loads the published candidate weight file (e.g. ``stf_0018_best.pth.tar`` or
``cnn_0018_best.pth.tar`` from the STF repo), translates it to compressai's
module layout, and writes a state dict that
``compressai.models.SymmetricalTransFormer.from_state_dict`` /
``compressai.models.WACNN.from_state_dict`` can load directly. Optionally
reports forward-pass sanity numbers (PSNR / bpp) on a synthetic input.

Example::

    python examples/convert_stf_checkpoint.py \\
        --src candidate/STF/stf_0018_best.pth.tar \\
        --arch stf \\
        --dst /tmp/stf_compressai.pth \\
        --smoke

    python examples/convert_stf_checkpoint.py \\
        --src candidate/STF/cnn_0018_best.pth.tar \\
        --arch wacnn \\
        --smoke
"""

from __future__ import annotations

import argparse

from pathlib import Path

import torch

from compressai.models.stf import (
    WACNN,
    SymmetricalTransFormer,
    convert_upstream_stf_state_dict,
)

_ARCHES = {"stf": SymmetricalTransFormer, "wacnn": WACNN}


def _detect_arch(state_dict: dict) -> str:
    keys = state_dict.keys()
    if any("patch_embed" in k for k in keys):
        return "stf"
    if any(k.endswith("g_a.0.weight") for k in keys):
        return "wacnn"
    raise SystemExit("could not auto-detect arch; pass --arch {stf,wacnn} explicitly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream checkpoint (e.g. stf_0018_best.pth.tar).",
    )
    parser.add_argument(
        "--arch",
        choices=sorted(_ARCHES),
        default=None,
        help="Architecture to instantiate. Auto-detected from key names if omitted.",
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
        help="Run a forward smoke test on a synthetic 256x256 image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"checkpoint not found: {args.src}")

    upstream = torch.load(args.src, map_location="cpu", weights_only=False)
    upstream = (
        upstream.get("state_dict", upstream) if isinstance(upstream, dict) else upstream
    )
    converted = convert_upstream_stf_state_dict(upstream)
    print(f"loaded {len(upstream)} upstream keys → {len(converted)} compressai keys")

    arch = args.arch or _detect_arch(upstream)
    cls = _ARCHES[arch]
    net = cls.from_state_dict(upstream)
    net.eval()
    print(f"loaded {arch.upper()}: {sum(p.numel() for p in net.parameters()):,} params")

    if args.dst is not None:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dst)
        print(f"wrote converted state dict → {args.dst}")

    if args.smoke:
        height = width = 256
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, height),
            torch.linspace(0, 1, width),
            indexing="ij",
        )
        img = (
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

        with torch.no_grad():
            out = net(img)
        n_pix = height * width
        psnr = -10 * torch.log10(((out["x_hat"].clamp(0, 1) - img) ** 2).mean()).item()
        y_bpp = -torch.log2(out["likelihoods"]["y"]).sum().item() / n_pix
        z_bpp = -torch.log2(out["likelihoods"]["z"]).sum().item() / n_pix
        print(
            f"smoke: PSNR={psnr:.2f}dB  y_bpp={y_bpp:.4f}  z_bpp={z_bpp:.4f}  "
            f"total_bpp={y_bpp + z_bpp:.4f}"
        )


if __name__ == "__main__":
    main()
