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
import re

from pathlib import Path
from typing import Dict

import torch

from torch import Tensor

from compressai.models.stf import WACNN, SymmetricalTransFormer

_ARCHES = {"stf": SymmetricalTransFormer, "wacnn": WACNN}


# ---------------------------------------------------------------------------
# Upstream STF / WACNN checkpoint conversion
#
# Lives here (not in compressai/models/stf.py) so the model module stays a
# clean compressai-native definition — `WACNN` / `SymmetricalTransFormer`
# `.from_state_dict` only load already-converted state dicts. Run this
# script once to translate a published upstream checkpoint into compressai
# layout, then load the result via `from_state_dict`.
# ---------------------------------------------------------------------------

_UPSTREAM_LATENT_CODEC_PREFIXES = (
    "cc_mean_transforms",
    "cc_scale_transforms",
    "lrp_transforms",
    "gaussian_conditional",
)

# Top-level rename map applied AFTER per-slice cc_/lrp_/gaussian_conditional
# rerooting. Keys are matched as exact prefixes (with the trailing dot).
_UPSTREAM_TOP_LEVEL_RENAMES: Dict[str, str] = {
    "h_a.": "latent_codec.h_a.",
    "h_mean_s.": "latent_codec.h_s.h_mean_s.",
    "h_scale_s.": "latent_codec.h_s.h_scale_s.",
    "entropy_bottleneck.": "latent_codec.z.entropy_bottleneck.",
}

# Upstream STF places the WindowAttention parameters directly under
# ``conv_b.<i>.attn.{qkv,proj,relative_position_*}``. CompressAI wraps the
# WindowAttention inside a :class:`compressai.layers.attn.swin.WMSA` shim, so
# the live model keeps ``WMSA.attn = WindowAttention(...)`` and the
# parameters land at ``conv_b.<i>.attn.attn.*``. This regex inserts the extra
# ``.attn`` so renamed upstream keys round-trip into the WMSA wrapper without
# changing the model topology.
_WMSA_NEST_PATTERN = re.compile(
    r"(\.conv_b\.\d+\.attn)\.(qkv\.|proj\.|relative_position_)"
)


def _nest_winmsa_keys(key: str) -> str:
    """Insert the WMSA wrapper level (``.attn``) into upstream
    ``conv_b.*.attn.{qkv,proj,relative_position_*}`` keys."""
    return _WMSA_NEST_PATTERN.sub(r"\1.attn.\2", key)


def _is_upstream_stf_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic: upstream checkpoints either carry a ``module.`` prefix or
    place ``cc_mean_transforms`` at the root instead of under ``latent_codec``.
    """
    for key in state_dict:
        if key.startswith("module."):
            return True
        head = key.split(".", 1)[0]
        if head in _UPSTREAM_LATENT_CODEC_PREFIXES or head in {
            "h_a",
            "h_mean_s",
            "h_scale_s",
            "entropy_bottleneck",
        }:
            return True
    return False


def convert_upstream_stf_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate a candidate ``STF`` / ``WACNN`` state dict into compressai layout.

    Upstream checkpoints (``stf_<bpp>_best.pth.tar`` / ``cnn_<bpp>_best.pth.tar``
    from `Zou et al. 2022 <https://arxiv.org/abs/2203.08450>`_) are saved from a
    ``DataParallel``-wrapped module and place the channel-conditional entropy
    transforms at the model root. After the H+G containerised refactor
    compressai houses those transforms (plus the Gaussian conditional and
    the hyperprior backbone) inside ``latent_codec.*``. This helper:

    - strips the leading ``module.`` prefix added by ``DataParallel``;
    - re-roots ``cc_mean_transforms.{k}`` / ``cc_scale_transforms.{k}`` /
      ``lrp_transforms.{k}`` under
      ``latent_codec.y.channel_context.y{k}.{mean_cc,scale_cc}.*`` /
      ``latent_codec.y.latent_codec.y{k}.lrp_transform.*``;
    - replicates the single shared ``gaussian_conditional.*`` buffer set
      under each per-slice leaf (``latent_codec.y.latent_codec.y{k}.gaussian_conditional.*``);
    - moves ``entropy_bottleneck.*`` / ``h_a.*`` / ``h_mean_s.*`` /
      ``h_scale_s.*`` under ``latent_codec.*`` per the new layout;
    - leaves ``g_a`` / ``g_s`` / ``patch_embed`` / ``layers`` /
      ``syn_layers`` / ``end_conv`` keys unchanged.

    The wiring sets ``emit_mean_support=True`` on the ``MeanScaleContextHead``
    so the upstream LRP layout (``cat(latent_means, *prev_y_hat, y_hat)``) is
    recoverable inside the leaf — upstream ``lrp_transforms.{k}`` weights
    transfer byte-for-byte. The model's ``WinNoShiftAttention`` consumers wrap
    their windowed-attention layers in a :class:`WMSA` shim, so the conversion
    also nests upstream ``conv_b.{i}.attn.{qkv,proj,relative_position_*}`` keys
    under the extra ``.attn`` level (see :func:`_nest_winmsa_keys`).

    The returned dict can be loaded directly by ``WACNN.from_state_dict`` /
    ``SymmetricalTransFormer.from_state_dict``.
    """
    converted: Dict[str, Tensor] = {}

    _LEGACY_ROOT_HEADS = set(_UPSTREAM_LATENT_CODEC_PREFIXES) | {
        "h_a",
        "h_mean_s",
        "h_scale_s",
        "entropy_bottleneck",
    }

    # Pass 1: strip module. prefix, fold the upstream single-``attn`` window
    # attention path back into compressai's WMSA wrapper layout, and inventory
    # which keys exist.
    cleaned: Dict[str, Tensor] = {}
    has_legacy_root_keys = False
    for key, value in state_dict.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        new_key = _nest_winmsa_keys(new_key)
        cleaned[new_key] = value
        if new_key.split(".", 1)[0] in _LEGACY_ROOT_HEADS:
            has_legacy_root_keys = True

    if not has_legacy_root_keys:
        # Already in (or near) the new layout — return cleaned dict as-is.
        return cleaned

    # Pass 2: discover slice indices to drive gaussian_conditional replication
    # and per-slice rerooting.
    slice_indices = sorted(
        {
            int(key.split(".")[1])
            for key in cleaned
            if key.startswith("cc_mean_transforms.")
        }
    )
    num_slices = len(slice_indices)

    for key, value in cleaned.items():
        head = key.split(".", 1)[0]
        if head == "cc_mean_transforms":
            _, k, *rest = key.split(".")
            new_key = f"latent_codec.y.channel_context.y{k}.mean_cc." + ".".join(rest)
            converted[new_key] = value
        elif head == "cc_scale_transforms":
            _, k, *rest = key.split(".")
            new_key = f"latent_codec.y.channel_context.y{k}.scale_cc." + ".".join(rest)
            converted[new_key] = value
        elif head == "lrp_transforms":
            _, k, *rest = key.split(".")
            new_key = f"latent_codec.y.latent_codec.y{k}.lrp_transform." + ".".join(
                rest
            )
            converted[new_key] = value
        elif head == "gaussian_conditional":
            # Replicate the single shared instance to per-slice leaves.
            tail = key[len("gaussian_conditional.") :]
            for k in range(num_slices):
                new_key = (
                    f"latent_codec.y.latent_codec.y{k}" f".gaussian_conditional.{tail}"
                )
                converted[new_key] = value
        else:
            renamed = key
            for prefix, replacement in _UPSTREAM_TOP_LEVEL_RENAMES.items():
                if key.startswith(prefix):
                    renamed = replacement + key[len(prefix) :]
                    break
            converted[renamed] = value

    return converted


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
    net = cls.from_state_dict(converted)
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
