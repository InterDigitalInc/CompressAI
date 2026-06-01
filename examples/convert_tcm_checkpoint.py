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
import re

from pathlib import Path
from typing import Dict, Tuple

import torch

from torch import Tensor

from compressai.models.tcm import TCM

# ----------------------------------------------------------------------------
# Upstream LIC_TCM checkpoint conversion
# ----------------------------------------------------------------------------


# Heads from upstream LIC_TCM (Liu et al. 2023) checkpoints that move under
# ``latent_codec.*`` after the H+G containerised refactor.
_UPSTREAM_LATENT_CODEC_PREFIXES = (
    "cc_mean_transforms",
    "cc_scale_transforms",
    "lrp_transforms",
    "atten_mean",
    "atten_scale",
    "mean_support_transforms",
    "scale_support_transforms",
    "gaussian_conditional",
)

# Top-level rename map applied AFTER per-slice rerooting. Keys are matched as
# exact prefixes (with the trailing dot).
_UPSTREAM_TOP_LEVEL_RENAMES: Dict[str, str] = {
    "h_a.": "latent_codec.h_a.",
    "h_mean_s.": "latent_codec.h_s.h_mean_s.",
    "h_scale_s.": "latent_codec.h_s.h_scale_s.",
    "entropy_bottleneck.": "latent_codec.z.entropy_bottleneck.",
}

# Upstream LIC_TCM wraps each ``SWAtten`` in an ``nn.Sequential`` and stores
# parameters at ``atten_mean.{k}.0.<...>``. Compressai's :class:`SWAtten`
# lives directly at ``mean_support_transform.<...>`` after rerooting, so the
# leading ``.0`` wrapper level is stripped.
_UPSTREAM_SWATTEN_WRAPPER = re.compile(
    r"^(atten_mean|atten_scale|mean_support_transforms|scale_support_transforms)\.(\d+)\.0\."
)


def _rename_msa_keys(key: str, value: Tensor) -> Tuple[str, Tensor]:
    """Translate upstream LIC_TCM ConvTransBlock-internal MSA layout to
    compressai's :class:`WMSA` wrapper layout.

    Three kinds of upstream keys appear inside ``g_a`` / ``g_s`` / ``h_a`` /
    ``h_mean_s`` / ``h_scale_s`` blocks:

    - ``.msa.relative_position_params`` is a ``(2*win-1, 2*win-1, num_heads)``
      buffer; compressai's ``WindowAttention`` registers it as a flat
      ``(N, num_heads)`` ``relative_position_bias_table``. The value is
      permuted and reshaped accordingly.
    - ``.msa.embedding_layer`` is upstream's name for the fused ``qkv``
      linear; compressai exposes it as ``.msa.attn.qkv.<...>``.
    - ``.msa.linear`` is upstream's optional output projection; compressai
      drops it and instead uses the WindowAttention's identity ``.proj`` —
      see :func:`_ensure_identity_attention_projection` for the identity
      injection that keeps strict ``load_state_dict`` round-trips clean.
    """
    if ".msa.relative_position_params" in key:
        new_key = key.replace(
            ".msa.relative_position_params",
            ".msa.attn.relative_position_bias_table",
        )
        new_value = value.permute(1, 2, 0).reshape(-1, value.size(0)).contiguous()
        return new_key, new_value
    if ".msa.embedding_layer." in key:
        return key.replace(".msa.embedding_layer.", ".msa.attn.qkv."), value
    if ".msa.linear." in key:
        return key.replace(".msa.linear.", ".msa.output_proj."), value
    return key, value


def _ensure_identity_attention_projection(
    state_dict: Dict[str, Tensor],
    output_proj_key: str,
    output_proj_value: Tensor,
) -> None:
    """Inject an identity ``WindowAttention.proj`` for upstream blocks whose
    output projection sits outside the attention module (``.msa.linear`` →
    ``.msa.output_proj``). The model has both ``.msa.attn.proj`` (inside
    WindowAttention, identity-initialised here) and ``.msa.output_proj``
    (the actual learned projection) so strict ``load_state_dict`` succeeds.
    """
    prefix, suffix = output_proj_key.rsplit(".msa.output_proj.", 1)
    attn_proj_key = f"{prefix}.msa.attn.proj.{suffix}"
    if attn_proj_key in state_dict:
        return
    if suffix == "weight":
        dimension = output_proj_value.size(0)
        state_dict[attn_proj_key] = torch.eye(
            dimension,
            dtype=output_proj_value.dtype,
            device=output_proj_value.device,
        )
        return
    if suffix == "bias":
        state_dict[attn_proj_key] = torch.zeros_like(output_proj_value)


def _is_upstream_tcm_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic: upstream LIC_TCM checkpoints either carry the ``module.``
    prefix from ``DataParallel`` saving, the ``.msa.relative_position_params``
    buffer, or the per-slice entropy heads (``cc_mean_transforms`` /
    ``atten_mean`` / ``lrp_transforms`` / ``gaussian_conditional`` / ``h_a``
    / ``h_mean_s`` / ``h_scale_s`` / ``entropy_bottleneck``) at the model
    root rather than under ``latent_codec.*``.
    """
    legacy_roots = set(_UPSTREAM_LATENT_CODEC_PREFIXES) | {
        "h_a",
        "h_mean_s",
        "h_scale_s",
        "entropy_bottleneck",
    }
    for key in state_dict:
        if key.startswith("module."):
            return True
        if (
            ".msa.relative_position_params" in key
            or ".msa.embedding_layer." in key
            or ".msa.linear." in key
        ):
            return True
        if key.split(".", 1)[0] in legacy_roots:
            return True
    return False


def convert_upstream_tcm_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream LIC_TCM state dict into compressai layout.

    Upstream checkpoints (e.g. ``0.013.pth..tar`` from
    `Liu et al. 2023 <https://arxiv.org/abs/2303.14978>`_,
    https://github.com/jmliu206/LIC_TCM) place the channel-conditional entropy
    transforms and the hyperprior backbone at the model root. After the H+G
    containerised refactor compressai houses those transforms (plus the
    Gaussian conditional and the ``z`` bottleneck) inside ``latent_codec.*``.
    This helper:

    - strips the leading ``module.`` prefix added by ``DataParallel``;
    - rewrites ConvTransBlock attention buffers via :func:`_rename_msa_keys`
      (``.msa.relative_position_params`` /  ``.msa.embedding_layer`` /
      ``.msa.linear``) and standard layer-name renames (``ln1`` → ``norm1``,
      ``mlp.0`` / ``mlp.2`` → ``mlp.fc1`` / ``mlp.fc2``);
    - unwraps the upstream ``nn.Sequential`` wrapper around each ``SWAtten``
      (``atten_mean.{k}.0.<...>`` → ``atten_mean.{k}.<...>``);
    - re-roots ``cc_mean_transforms.{k}`` / ``cc_scale_transforms.{k}`` /
      ``lrp_transforms.{k}`` under
      ``latent_codec.y.channel_context.y{k}.{mean_cc,scale_cc}.*`` /
      ``latent_codec.y.latent_codec.y{k}.lrp_transform.*``;
    - re-roots ``atten_mean.{k}`` / ``atten_scale.{k}`` (or their
      ``mean_support_transforms`` / ``scale_support_transforms`` aliases)
      under ``latent_codec.y.channel_context.y{k}.{mean,scale}_support_transform.*``;
    - replicates the single shared ``gaussian_conditional.*`` buffer set
      under each per-slice leaf
      (``latent_codec.y.latent_codec.y{k}.gaussian_conditional.*``);
    - moves ``entropy_bottleneck.*`` / ``h_a.*`` / ``h_mean_s.*`` /
      ``h_scale_s.*`` under ``latent_codec.*`` per the new layout;
    - leaves ``g_a`` / ``g_s`` keys (other than the MSA renames inside their
      ConvTransBlocks) untouched.

    The wiring sets ``emit_mean_support=True`` on the
    :class:`MeanScaleContextHead`, so the upstream LRP layout
    (``cat(latent_means, *prev_y_hat, y_hat)``) is recoverable inside the
    leaf — upstream ``lrp_transforms.{k}`` weights therefore transfer
    byte-for-byte.

    The returned dict can be loaded by :meth:`TCM.from_state_dict`, which
    auto-detects the upstream layout and calls this helper, so direct
    invocation is only needed when persisting the converted dict.
    """
    # Pass 1: strip ``module.`` prefix; rewrite ConvTransBlock attention
    # buffers and layer names; unwrap the SWAtten ``nn.Sequential`` wrapper;
    # alias ``atten_mean`` / ``atten_scale`` to the canonical
    # ``mean_support_transforms`` / ``scale_support_transforms`` names so the
    # per-slice rerooting in Pass 2 only has to handle one form.
    cleaned: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        new_key, value = _rename_msa_keys(new_key, value)
        wrapper = _UPSTREAM_SWATTEN_WRAPPER.match(new_key)
        if wrapper:
            new_key = (
                f"{wrapper.group(1)}.{wrapper.group(2)}." + new_key[wrapper.end() :]
            )
        if new_key.startswith("atten_mean."):
            new_key = "mean_support_transforms." + new_key[len("atten_mean.") :]
        elif new_key.startswith("atten_scale."):
            new_key = "scale_support_transforms." + new_key[len("atten_scale.") :]
        new_key = new_key.replace(".ln1.", ".norm1.")
        new_key = new_key.replace(".ln2.", ".norm2.")
        new_key = new_key.replace(".mlp.0.", ".mlp.fc1.")
        new_key = new_key.replace(".mlp.2.", ".mlp.fc2.")
        if ".msa.output_proj." in new_key:
            _ensure_identity_attention_projection(cleaned, new_key, value)
        cleaned[new_key] = value

    # Pass 2: discover slice indices to drive ``gaussian_conditional``
    # replication, then reroot per-slice and top-level keys.
    converted: Dict[str, Tensor] = {}
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
        elif head == "mean_support_transforms":
            _, k, *rest = key.split(".")
            new_key = (
                f"latent_codec.y.channel_context.y{k}.mean_support_transform."
                + ".".join(rest)
            )
            converted[new_key] = value
        elif head == "scale_support_transforms":
            _, k, *rest = key.split(".")
            new_key = (
                f"latent_codec.y.channel_context.y{k}.scale_support_transform."
                + ".".join(rest)
            )
            converted[new_key] = value
        elif head == "lrp_transforms":
            _, k, *rest = key.split(".")
            new_key = f"latent_codec.y.latent_codec.y{k}.lrp_transform." + ".".join(
                rest
            )
            converted[new_key] = value
        elif head == "gaussian_conditional":
            tail = key[len("gaussian_conditional.") :]
            for k in range(num_slices):
                new_key = (
                    f"latent_codec.y.latent_codec.y{k}.gaussian_conditional.{tail}"
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

    net = TCM.from_state_dict(converted)
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
