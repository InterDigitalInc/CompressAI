"""Convert an upstream CCA checkpoint to compressai layout.

Loads the published candidate weight file (e.g.
``checkpoint_lambda_0.3.pth.tar`` from M. Han et al.,
https://github.com/CVL-UESTC/CCA, NeurIPS 2024), translates it to
compressai's module layout, and writes a state dict that
``compressai.models.cca.CCAModel.from_state_dict`` can load directly.
Optionally reports forward-pass sanity numbers (PSNR / bpp) on a
synthetic input.

The upstream-vs-compressai key differences (NAFBlock interior renames,
``mean_NAF_transforms`` -> ``channel_context.y{k}.mean_support_transform``,
``mean_cc_transforms.{k}`` -> ``channel_context.y{k}.mean_cc``,
``lrp_transforms.{k}`` -> ``latent_codec.y{k}.lrp_transform``,
``aux_entropymodel.*`` -> ``aux_entropy_model.inner_codec.*``, the
gaussian_conditional replication across slices, the H+G containerised
re-rooting under ``latent_codec.*``, etc.) are all handled inside
``convert_upstream_cca_state_dict``; this script is a thin CLI around it.

Example::

    python examples/convert_cca_checkpoint.py \\
        --src candidate/CCA/checkpoint_lambda_0.3.pth.tar \\
        --dst /tmp/cca_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse

from pathlib import Path

import torch

from compressai.models.cca import CCAModel, convert_upstream_cca_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream CCA checkpoint (e.g. checkpoint_lambda_0.3.pth.tar).",
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
    converted = convert_upstream_cca_state_dict(upstream)
    print(f"loaded {len(upstream)} upstream keys → {len(converted)} compressai keys")

    net = CCAModel.from_state_dict(upstream)
    net.eval()
    print(
        "variant: "
        f"M={net.M}, N={net.N}, slice_sizes={net.slice_sizes}, "
        f"em_hidden={net.em_hidden_channels}, em_layers={net.em_num_layers}, "
        f"cca_training={net.cca_training}"
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
