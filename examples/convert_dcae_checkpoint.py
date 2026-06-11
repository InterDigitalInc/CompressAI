"""Convert an upstream DCAE checkpoint to compressai layout.

Loads the published DCAE weight file (e.g. ``0.0018checkpoint_best.pth.tar``
from the upstream DCAE release accompanying Lu et al., CVPR 2025), translates
it to compressai's containerized module layout, and writes a state dict that
``compressai.models.dcae.DCAE.from_state_dict`` can load directly. Optionally
reports forward-pass sanity numbers (PSNR / bpp) on a synthetic input.

The upstream-vs-compressai key differences (top-level ``dt`` -> shared
dictionary submodule, ``cc_mean_transforms.{k}`` /
``cc_scale_transforms.{k}`` / ``lrp_transforms.{k}`` /
``dt_cross_attention.{k}`` ModuleLists -> per-slice
``latent_codec.y.channel_context.y{k}`` / ``latent_codec.y.latent_codec.y{k}``
entries, the means/scales swap on the leading 2*M input channels of the
first conv / linear weights, the ``h_z_s2``/``h_z_s1`` -> means/scales
swap, the H+G containerized re-rooting under ``latent_codec.*``, etc.)
are all handled inside ``convert_upstream_dcae_state_dict``; this script
is a thin CLI around it.

Example::

    python examples/convert_dcae_checkpoint.py \\
        --src candidate/DCAE/0.0018checkpoint_best.pth.tar \\
        --dst /tmp/dcae_compressai.pth \\
        --smoke
"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Dict

import torch

from torch import Tensor

from compressai.models.dcae import DCAE

# ----------------------------------------------------------------------------
# Upstream → compressai state-dict conversion.
#
# Lives here (not in compressai/models/dcae.py) so the model module stays a
# clean compressai-native definition — ``DCAE.from_state_dict`` only loads
# already-converted state dicts. Run this script once to translate a
# published upstream checkpoint into compressai layout, then load the result
# via ``from_state_dict``.
# ----------------------------------------------------------------------------


def _is_upstream_layout(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic: upstream DCAE has top-level ``dt`` and ``dt_cross_attention.0.*``
    (with or without a ``module.`` ``DataParallel`` prefix)."""
    return any(key in ("dt", "module.dt") for key in state_dict) and any(
        k.startswith("dt_cross_attention.")
        or k.startswith("module.dt_cross_attention.")
        for k in state_dict
    )


def _swap_first_2m_in_channels(weight: Tensor, m: int) -> Tensor:
    """Swap the first ``m`` and second ``m`` slices along ``dim=1``.

    Upstream DCAE assembles its query as
    ``cat([latent_scales(m), latent_means(m), *prev_y_hat])`` (scales first),
    whereas the containerized wiring uses ``cat([latent_means(m),
    latent_scales(m), *prev_y_hat])`` (means first). Permuting the leading
    2*m input channels of the first conv / linear weight in cc_mean,
    cc_scale, lrp_transforms, and cross_attention.x_trans rebases the
    upstream weights to the new input order with no retraining.
    """
    if weight.dim() < 2 or weight.size(1) < 2 * m:
        return weight
    permuted = weight.clone()
    permuted[:, :m] = weight[:, m : 2 * m]
    permuted[:, m : 2 * m] = weight[:, :m]
    return permuted


def convert_upstream_dcae_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Convert an upstream DCAE checkpoint to the containerized layout.

    The upstream DCAE source follows the master-era monolithic pattern
    (model owns ``h_a`` / ``h_z_s1`` / ``h_z_s2`` / ``entropy_bottleneck`` /
    ``gaussian_conditional`` / per-slice ModuleLists for cc heads / LRP /
    cross-attention plus a top-level ``dt`` Parameter). This function
    rewrites the keys to the containerized layout used by
    :class:`~compressai.models.dcae.DCAE` post-refactor:

    - ``dt`` -> ``shared_dictionary.dt``
    - ``dt_cross_attention.{k}.*`` ->
      ``latent_codec.y.channel_context.y{k}.cross_attention.*``
    - ``cc_mean_transforms.{k}.*`` ->
      ``latent_codec.y.channel_context.y{k}.mean_cc.*``
    - ``cc_scale_transforms.{k}.*`` ->
      ``latent_codec.y.channel_context.y{k}.scale_cc.*``
    - ``lrp_transforms.{k}.*`` ->
      ``latent_codec.y.latent_codec.y{k}.lrp_transform.*``
    - ``gaussian_conditional.*`` -> fanned out to K copies under
      ``latent_codec.y.latent_codec.y{k}.gaussian_conditional.*``
    - ``h_a.*`` -> ``latent_codec.h_a.*``
    - ``h_z_s2.*`` -> ``latent_codec.h_s.h_mean_s.*`` (means)
    - ``h_z_s1.*`` -> ``latent_codec.h_s.h_scale_s.*`` (scales)
    - ``entropy_bottleneck.*`` -> ``latent_codec.z.entropy_bottleneck.*``

    Additionally permutes the leading 2*M input channels of the first
    conv / linear weight in ``cc_mean`` / ``cc_scale`` / ``lrp_transform`` /
    ``cross_attention.x_trans`` to swap upstream's
    ``[scales, means, ...]`` ordering to the containerized
    ``[means, scales, ...]`` ordering — see :func:`_swap_first_2m_in_channels`.
    """
    if not _is_upstream_layout(state_dict):
        return state_dict

    # Strip the ``module.`` ``DataParallel`` prefix if present.
    state_dict = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    # First pass: figure out M and num_slices from the input dict so we know
    # how many cc / lrp / cross_attention slots to fan out and how to size
    # the means/scales swap.
    if "h_a.0.conv.weight" in state_dict:
        m = state_dict["h_a.0.conv.weight"].size(1)
    else:
        # Fallback: gaussian_conditional doesn't expose M; rely on dt_cross_attention.0.linear.weight
        m = state_dict["dt_cross_attention.0.linear.weight"].size(0)
    cc_indices = sorted(
        {
            int(k.split(".")[1])
            for k in state_dict
            if k.startswith("cc_mean_transforms.")
        }
    )
    if not cc_indices:
        raise ValueError("cannot infer num_slices from upstream state_dict")
    num_slices = max(cc_indices) + 1

    out: Dict[str, Tensor] = {}

    # Helper to rewrite a per-slice ModuleList key like "cc_mean_transforms.0.0.weight"
    # to a containerized destination, optionally swapping first-2M channels in the
    # first sub-module (index "0").
    def _reroot_modlist(
        src_prefix: str, dst_prefix: str, *, swap_first_conv: bool
    ) -> None:
        for key, value in state_dict.items():
            if not key.startswith(src_prefix):
                continue
            tail = key[
                len(src_prefix) :
            ]  # e.g., "0.0.weight" (slice 0, sub 0, "weight")
            parts = tail.split(".", 2)
            if len(parts) < 2:
                continue
            slice_idx, sub_idx = parts[0], parts[1]
            sub_tail = parts[2] if len(parts) > 2 else ""
            new_value = value
            if swap_first_conv and sub_idx == "0" and sub_tail == "weight":
                new_value = _swap_first_2m_in_channels(value, m)
            dst_key = f"{dst_prefix.format(slice_idx=slice_idx)}.{sub_idx}"
            if sub_tail:
                dst_key = f"{dst_key}.{sub_tail}"
            out[dst_key] = new_value

    _reroot_modlist(
        "cc_mean_transforms.",
        "latent_codec.y.channel_context.y{slice_idx}.mean_cc",
        swap_first_conv=True,
    )
    _reroot_modlist(
        "cc_scale_transforms.",
        "latent_codec.y.channel_context.y{slice_idx}.scale_cc",
        swap_first_conv=True,
    )
    _reroot_modlist(
        "lrp_transforms.",
        "latent_codec.y.latent_codec.y{slice_idx}.lrp_transform",
        swap_first_conv=True,
    )

    # cross_attention is a ModuleList of MutiScaleDictionaryCrossAttentionGLU; its
    # x_trans.weight is the only Linear that consumes the [scales, means, ...] input.
    for key, value in state_dict.items():
        if not key.startswith("dt_cross_attention."):
            continue
        # Tail layout: "{slice_idx}.{remainder}".
        tail = key[len("dt_cross_attention.") :]
        slice_idx, remainder = tail.split(".", 1)
        new_value = value
        if remainder == "x_trans.weight":
            new_value = _swap_first_2m_in_channels(value, m)
        dst_key = (
            f"latent_codec.y.channel_context.y{slice_idx}.cross_attention.{remainder}"
        )
        out[dst_key] = new_value

    # Fan out the single shared gaussian_conditional to K per-slice copies.
    for key, value in state_dict.items():
        if not key.startswith("gaussian_conditional."):
            continue
        suffix = key[len("gaussian_conditional.") :]
        for slice_idx in range(num_slices):
            out[
                f"latent_codec.y.latent_codec.y{slice_idx}.gaussian_conditional.{suffix}"
            ] = value

    # Top-level renames.
    top_level_renames = {
        "dt": "shared_dictionary.dt",
        "h_a.": "latent_codec.h_a.",
        "h_z_s1.": "latent_codec.h_s.h_scale_s.",
        "h_z_s2.": "latent_codec.h_s.h_mean_s.",
        "entropy_bottleneck.": "latent_codec.z.entropy_bottleneck.",
    }
    for key, value in state_dict.items():
        if key in ("dt",) or key.startswith(
            (
                "h_a.",
                "h_z_s1.",
                "h_z_s2.",
                "entropy_bottleneck.",
            )
        ):
            for src, dst in top_level_renames.items():
                if key == src:
                    out[dst] = value
                    break
                if src.endswith(".") and key.startswith(src):
                    out[dst + key[len(src) :]] = value
                    break
        elif key.startswith(
            (
                "cc_mean_transforms.",
                "cc_scale_transforms.",
                "lrp_transforms.",
                "dt_cross_attention.",
                "gaussian_conditional.",
            )
        ):
            # Already handled above.
            continue
        else:
            # Carry-over keys not part of the entropy / hyperprior sections
            # (e.g., g_a / g_s parameters).
            out[key] = value

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the upstream DCAE checkpoint (e.g. 0.0018checkpoint_best.pth.tar).",
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
    if _is_upstream_layout(upstream):
        converted = convert_upstream_dcae_state_dict(upstream)
    else:
        converted = upstream
    print(f"loaded {len(upstream)} upstream keys → {len(converted)} compressai keys")

    net = DCAE.from_state_dict(converted)
    net.eval()
    print(
        "variant: "
        f"N={net.N}, M={net.M}, hyper_channels={net.hyper_channels}, "
        f"num_slices={net.num_slices}, max_support_slices={net.max_support_slices}, "
        f"feature_dims={tuple(net.feature_dims)}, block_num={tuple(net.block_num)}, "
        f"dict_num={net.dict_num}, dict_head_num={net.dict_head_num}, "
        f"dictionary_dim={net.dictionary_dim}"
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
