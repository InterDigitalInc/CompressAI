"""Convert an upstream CCA checkpoint to compressai layout.

Loads the published candidate weight file (e.g.
``checkpoint_lambda_0.3.pth.tar`` from M. Han et al.,
https://github.com/CVL-UESTC/CCA, NeurIPS 2024), translates it to
compressai's module layout, and writes a state dict that
``compressai.models.cca.CCAModel.from_state_dict`` can load directly.
Optionally reports forward-pass sanity numbers (PSNR / bpp) on a
synthetic input.

The upstream-vs-compressai key differences (NAFBlock interior renames
``dwconv`` / ``sca`` / ``FFN`` / ``conv1``, NAFTransform ``in_conv`` /
``out_conv``, ``mean_NAF_transforms.{k}`` ->
``channel_context.y{k}.mean_support_transform``, ``mean_cc_transforms.{k}``
-> ``channel_context.y{k}.mean_cc``, ``lrp_transforms.{k}`` ->
``latent_codec.y{k}.lrp_transform``, ``aux_entropymodel.*`` ->
``aux_entropy_model.inner_codec.*``, the ``gaussian_conditional``
replication across slices, the H+G containerised re-rooting under
``latent_codec.*``, etc.) are all handled inside
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
from typing import Dict, List, Optional, Sequence

import torch

from torch import Tensor

from compressai.models.cca import CCAModel

# ----------------------------------------------------------------------------
# Upstream → compressai state-dict conversion.
#
# Lives here (not in compressai/models/cca.py) so the model module stays a
# clean compressai-native definition — ``CCAModel.from_state_dict`` only
# loads already-converted state dicts. Run this script once to translate a
# published upstream checkpoint into compressai layout, then load the result
# via ``from_state_dict``.
# ----------------------------------------------------------------------------


# NAFBlock interior renames (upstream -> compressai). These are scoped to
# detected NAFBlock prefixes so they don't accidentally rewrite ``conv1`` in
# unrelated modules (e.g. ResidualBottleneckBlock has its own ``conv1``).
_NAF_BLOCK_RENAMES = {
    "dwconv.": "pointwise_depthwise.",
    "sca.": "channel_attention.",
    "FFN.": "feed_forward.",
    "conv1.": "project.",
}
# NAFTransform interior renames.
_NAF_TRANSFORM_RENAMES = {
    "in_conv.": "input_projection.",
    "out_conv.": "output_projection.",
}
# Top-level rename map applied AFTER NAFBlock / NAFTransform interior renames
# and BEFORE per-slice rerooting. Used for hyperprior backbone and aux module.
_TOPLEVEL_RENAMES: Dict[str, str] = {
    "aux_entropymodel.": "aux_entropy_model.",
    "h_a.": "latent_codec.h_a.",
    "h_mean_s.": "latent_codec.h_s.h_mean_s.",
    "h_scale_s.": "latent_codec.h_s.h_scale_s.",
    "z_entropy_bottleneck.": "latent_codec.z.entropy_bottleneck.",
}
# Upstream uses ``mean_NAF_transforms`` / ``scale_NAF_transforms``; compressai
# stores them at ``{mean,scale}_support_transform`` inside the channel-context
# head (singular per slice). Aliasing here keeps the per-slice rerooting pass
# uniform across main and aux branches.
_NAMED_PART_RENAMES: Dict[str, str] = {
    "mean_NAF_transforms.": "mean_support_transforms.",
    "scale_NAF_transforms.": "scale_support_transforms.",
}


def _is_upstream_cca_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic detector for upstream ``LICAutoencoder`` checkpoints."""
    for key in state_dict:
        if (
            key.startswith("mean_NAF_transforms.")
            or key.startswith("scale_NAF_transforms.")
            or key.startswith("aux_entropymodel.")
            or key.startswith("z_entropy_bottleneck.")
            or key.startswith("mean_cc_transforms.")
            or key.startswith("scale_cc_transforms.")
            or key.startswith("lrp_transforms.")
        ):
            return True
    return False


def _find_naf_block_prefixes(state_dict: Dict[str, Tensor]) -> List[str]:
    """Locate every NAFBlock instance by matching the ``.beta`` /  ``.gamma``
    / ``.dwconv.0.weight`` / ``.FFN.0.weight`` 4-tuple at the same scope.
    """
    suffix = ".beta"
    out: List[str] = []
    for key in state_dict:
        if not key.endswith(suffix):
            continue
        base = key[: -len(suffix)]
        if (
            f"{base}.gamma" in state_dict
            and f"{base}.dwconv.0.weight" in state_dict
            and f"{base}.FFN.0.weight" in state_dict
        ):
            out.append(base)
    return out


def _find_naf_transform_prefixes(state_dict: Dict[str, Tensor]) -> List[str]:
    """Locate every NAFTransform instance by matching the ``.in_conv.weight``
    / ``.out_conv.weight`` / ``.blocks.0.beta`` triple at the same scope.
    """
    suffix = ".in_conv.weight"
    out: List[str] = []
    for key in state_dict:
        if not key.endswith(suffix):
            continue
        base = key[: -len(suffix)]
        if (
            f"{base}.out_conv.weight" in state_dict
            and f"{base}.blocks.0.beta" in state_dict
        ):
            out.append(base)
    return out


def _strip_prefix(key: str, prefix: str) -> Optional[str]:
    return key[len(prefix) :] if key.startswith(prefix) else None


def _rename_with_table(
    key: str,
    base_prefixes: Sequence[str],
    rename_map: Dict[str, str],
) -> str:
    for base in base_prefixes:
        head = base + "."
        rest = _strip_prefix(key, head)
        if rest is None:
            continue
        for old, new in rename_map.items():
            inner = _strip_prefix(rest, old)
            if inner is not None:
                return head + new + inner
        return key
    return key


def _reroot_per_slice_keys(
    cleaned: Dict[str, Tensor],
    converted: Dict[str, Tensor],
    *,
    legacy_prefix: str,
    container_prefix: str,
    sub_name: str,
    num_slices: int,
    consume: List[str],
) -> None:
    """Move ``legacy_prefix.{k}.<...>`` keys to
    ``container_prefix.y{k}.sub_name.<...>``.

    Keys that match are removed from ``cleaned`` (recorded in ``consume``
    for a later bulk drop) and inserted into ``converted`` under the new
    path.
    """
    for key in list(cleaned.keys()):
        rest = _strip_prefix(key, legacy_prefix + ".")
        if rest is None:
            continue
        idx_str, _, tail = rest.partition(".")
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        if idx >= num_slices:
            continue
        new_key = (
            f"{container_prefix}.y{idx}.{sub_name}.{tail}"
            if tail
            else f"{container_prefix}.y{idx}.{sub_name}"
        )
        converted[new_key] = cleaned[key]
        consume.append(key)


def _replicate_gaussian_conditional(
    cleaned: Dict[str, Tensor],
    converted: Dict[str, Tensor],
    *,
    legacy_prefix: str,
    new_prefix: str,
    num_slices: int,
    consume: List[str],
) -> None:
    """Copy a single shared ``gaussian_conditional.<...>`` buffer set under
    every per-slice leaf so the per-slice
    :class:`GaussianConditionalLatentCodec` copies all strict-load.
    """
    for key in list(cleaned.keys()):
        tail = _strip_prefix(key, legacy_prefix + ".")
        if tail is None:
            continue
        for k in range(num_slices):
            new_key = f"{new_prefix}.y{k}.gaussian_conditional.{tail}"
            converted[new_key] = cleaned[key]
        consume.append(key)


def convert_upstream_cca_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream CCA ``LICAutoencoder`` state dict to the
    compressai layout produced by :class:`CCAModel`.

    Conversion runs three logical passes:

    1. Interior renames: ``NAFBlock`` (``dwconv`` → ``pointwise_depthwise``,
       etc.) and ``NAFTransform`` (``in_conv`` → ``input_projection``,
       etc.). Detection is by structural fingerprint
       (:func:`_find_naf_block_prefixes`) so the renames apply uniformly to
       NAFBlocks anywhere in the state dict (``g_a`` / ``g_s`` / per-slice
       support transforms / aux module).
    2. Top-level renames: ``aux_entropymodel`` → ``aux_entropy_model``,
       hyperprior backbone (``h_a`` / ``h_mean_s`` / ``h_scale_s``) and
       ``z_entropy_bottleneck`` are moved under ``latent_codec.*``;
       ``mean_NAF_transforms`` / ``scale_NAF_transforms`` are aliased to
       the singular ``{mean,scale}_support_transforms`` form so the
       per-slice rerooting in pass 3 only handles one name.
    3. Per-slice rerooting: ``mean_cc_transforms.{k}`` /
       ``scale_cc_transforms.{k}`` move to
       ``latent_codec.y.channel_context.y{k}.{mean,scale}_cc.*``;
       ``mean_support_transforms.{k}`` / ``scale_support_transforms.{k}``
       move to
       ``latent_codec.y.channel_context.y{k}.{mean,scale}_support_transform.*``;
       ``lrp_transforms.{k}`` moves to
       ``latent_codec.y.latent_codec.y{k}.lrp_transform.*``; the single
       shared ``gaussian_conditional.*`` buffer set is replicated under
       every per-slice leaf
       (``latent_codec.y.latent_codec.y{k}.gaussian_conditional.*``). The
       same rerooting is applied to ``aux_entropy_model.*`` (after the
       top-level rename) under ``aux_entropy_model.inner_codec.*``.

    The returned dict can be loaded by :meth:`CCAModel.from_state_dict`.
    """
    naf_blocks = _find_naf_block_prefixes(state_dict)
    naf_transforms = _find_naf_transform_prefixes(state_dict)

    # Pass 1+2: interior + top-level renames.
    cleaned: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = _rename_with_table(key, naf_blocks, _NAF_BLOCK_RENAMES)
        new_key = _rename_with_table(new_key, naf_transforms, _NAF_TRANSFORM_RENAMES)
        for old, new in _NAMED_PART_RENAMES.items():
            new_key = new_key.replace(old, new)
        for old, new in _TOPLEVEL_RENAMES.items():
            if new_key.startswith(old):
                new_key = new + new_key[len(old) :]
                break
        cleaned[new_key] = value

    # Pass 3a: per-slice rerooting for the main entropy stack. Discover
    # ``num_slices`` from ``mean_cc_transforms`` first, then drive the rest.
    main_indices = sorted(
        {
            int(key[len("mean_cc_transforms.") :].split(".", 1)[0])
            for key in cleaned
            if key.startswith("mean_cc_transforms.")
        }
    )
    num_slices_main = len(main_indices)

    converted: Dict[str, Tensor] = {}
    consumed: List[str] = []

    if num_slices_main:
        for legacy, container, sub in (
            ("mean_cc_transforms", "latent_codec.y.channel_context", "mean_cc"),
            ("scale_cc_transforms", "latent_codec.y.channel_context", "scale_cc"),
            (
                "mean_support_transforms",
                "latent_codec.y.channel_context",
                "mean_support_transform",
            ),
            (
                "scale_support_transforms",
                "latent_codec.y.channel_context",
                "scale_support_transform",
            ),
            ("lrp_transforms", "latent_codec.y.latent_codec", "lrp_transform"),
        ):
            _reroot_per_slice_keys(
                cleaned,
                converted,
                legacy_prefix=legacy,
                container_prefix=container,
                sub_name=sub,
                num_slices=num_slices_main,
                consume=consumed,
            )
        _replicate_gaussian_conditional(
            cleaned,
            converted,
            legacy_prefix="gaussian_conditional",
            new_prefix="latent_codec.y.latent_codec",
            num_slices=num_slices_main,
            consume=consumed,
        )

    # Pass 3b: per-slice rerooting inside the aux entropy module. Discover
    # ``num_slices_aux`` from ``aux_entropy_model.mean_cc_transforms``.
    aux_indices = sorted(
        {
            int(key[len("aux_entropy_model.mean_cc_transforms.") :].split(".", 1)[0])
            for key in cleaned
            if key.startswith("aux_entropy_model.mean_cc_transforms.")
        }
    )
    num_slices_aux = len(aux_indices)
    if num_slices_aux:
        for legacy, container, sub in (
            (
                "aux_entropy_model.mean_cc_transforms",
                "aux_entropy_model.inner_codec.channel_context",
                "mean_cc",
            ),
            (
                "aux_entropy_model.scale_cc_transforms",
                "aux_entropy_model.inner_codec.channel_context",
                "scale_cc",
            ),
            (
                "aux_entropy_model.mean_support_transforms",
                "aux_entropy_model.inner_codec.channel_context",
                "mean_support_transform",
            ),
            (
                "aux_entropy_model.scale_support_transforms",
                "aux_entropy_model.inner_codec.channel_context",
                "scale_support_transform",
            ),
            (
                "aux_entropy_model.lrp_transforms",
                "aux_entropy_model.inner_codec.latent_codec",
                "lrp_transform",
            ),
        ):
            _reroot_per_slice_keys(
                cleaned,
                converted,
                legacy_prefix=legacy,
                container_prefix=container,
                sub_name=sub,
                num_slices=num_slices_aux,
                consume=consumed,
            )
        _replicate_gaussian_conditional(
            cleaned,
            converted,
            legacy_prefix="aux_entropy_model.gaussian_conditional",
            new_prefix="aux_entropy_model.inner_codec.latent_codec",
            num_slices=num_slices_aux,
            consume=consumed,
        )

    for key in consumed:
        cleaned.pop(key, None)
    # Remaining keys (g_a / g_s / latent_codec.* hyperprior backbone /
    # aux_entropy_model.y_entropy_bottleneck / etc.) pass through unchanged.
    converted.update(cleaned)
    return converted


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
    if _is_upstream_cca_state_dict(upstream):
        converted = convert_upstream_cca_state_dict(upstream)
    else:
        converted = upstream
    print(f"loaded {len(upstream)} upstream keys → {len(converted)} compressai keys")

    net = CCAModel.from_state_dict(converted)
    net.eval()
    print(
        "variant: "
        f"M={net.M}, N={net.N}, slice_sizes={tuple(net.slice_sizes)}, "
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
