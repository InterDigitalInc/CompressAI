"""Convert an upstream LIC-TCM checkpoint to compressai layout.

Loads the published candidate weight file (e.g. ``0.05.pth.tar`` or
``mse_lambda_0.05.pth.tar`` from the LIC_TCM repo,
https://github.com/jmliu206/LIC_TCM), translates it to compressai's module
layout, and writes a state dict that ``compressai.models.tcm.TCM.from_state_dict``
can load directly. Optionally reports forward-pass sanity numbers
(PSNR / bpp) on a synthetic input.

The upstream-vs-compressai key differences (``module.`` ``DataParallel``
prefix, the ``nn.Sequential`` wrapper around each ``SWAtten``,
``atten_mean`` -> ``latent_codec.y.channel_context.y{k}.mean_support_transform``,
ConvTransBlock MSA buffer layouts, layer-norm names, the H+G containerised
re-rooting under ``latent_codec.*``, etc.) are all handled inside
``convert_upstream_tcm_state_dict``; this script is a thin CLI around it.

Example::

    python examples/convert_tcm_checkpoint.py \\
        --src candidate/TCM/0.05.pth.tar \\
        --dst /tmp/tcm_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse

from pathlib import Path

import torch

from compressai.models.tcm import TCM, convert_upstream_tcm_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream LIC-TCM checkpoint (e.g. 0.05.pth.tar).",
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
    converted = convert_upstream_tcm_state_dict(upstream)
    print(f"loaded {len(upstream)} upstream keys → {len(converted)} compressai keys")

    net = TCM.from_state_dict(upstream)
    net.eval()
    print(
        "variant: "
        f"N={net.N}, M={net.M}, num_slices={net.num_slices}, "
        f"config={tuple(net.config)}, head_dim={tuple(net.head_dim)}, "
        f"hyper_channels={net.hyper_channels}, "
        f"max_support_slices={net.max_support_slices}"
    )
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

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
