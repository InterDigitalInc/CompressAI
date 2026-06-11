"""AuxT (Auxiliary Transform) primitives + model-integration helpers.

Z. Li et al., "On Disentangled Training for Nonlinear Transform in Learned
Image Compression", ICLR 2025 (Spotlight,
https://arxiv.org/abs/2501.13751).

This module consolidates everything model-side that is AuxT-specific:

- :class:`OLP` — Orthogonal Linear Projection primitive (no extra
  dependency). Used both as a standalone channel mixer (SAAF
  ``_AdaptiveFrequencyBlock``) and inside :class:`WLS` / :class:`iWLS`.
- :class:`WLS` / :class:`iWLS` — wavelet-based analysis / synthesis
  blocks pairing a :class:`~compressai.layers.wave.DWT2D` /
  :class:`~compressai.layers.wave.IDWT2D` with learnable per-subband
  scaling and an :class:`OLP` channel mixer. Lazily imports
  :mod:`compressai.layers.wave` so :func:`aux_loss` and the side-branch
  helpers below stay importable without ``pytorch_wavelets``.
- Side-branch builders + walker (:func:`build_wls_branch`,
  :func:`build_iwls_branch`, :func:`forward_with_auxt`,
  :func:`compute_analysis_aux_positions`,
  :func:`compute_synthesis_aux_positions`) for hosts that integrate AuxT
  as a parallel chain summed into ``g_a`` / ``g_s`` (TCM ``use_auxt`` and
  any future model with the same six-stage config).
- :func:`aux_loss` — generic OLP regulariser aggregator used by both
  TCM-style (side-branch) and SAAF-style (integral) hosts.
- State-dict utilities (:func:`has_auxt_state`,
  :func:`is_auxt_wavelet_buffer_key`,
  :func:`is_auxt_upstream_wavelet_buffer_key`,
  :func:`normalize_upstream_auxt_key`) that any host's
  ``from_state_dict`` / ``convert_upstream_*_state_dict`` can reuse.

The wavelet-only :class:`~compressai.layers.wave.DWT2D` /
:class:`~compressai.layers.wave.IDWT2D` wrappers are kept under
:mod:`compressai.layers.wave` because they are generic
``pytorch_wavelets`` adapters that future non-AuxT models (e.g.
WeConvene) may want to reuse.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

__all__ = [
    "OLP",
    "WLS",
    "aux_loss",
    "build_iwls_branch",
    "build_wls_branch",
    "compute_analysis_aux_positions",
    "compute_synthesis_aux_positions",
    "forward_with_auxt",
    "has_auxt_state",
    "is_auxt_upstream_wavelet_buffer_key",
    "is_auxt_wavelet_buffer_key",
    "iWLS",
    "normalize_upstream_auxt_key",
]


# ---------------------------------------------------------------------------
# AuxT primitives — OLP, WLS, iWLS
# ---------------------------------------------------------------------------


class OLP(nn.Module):
    """Orthogonal linear projection with an auxiliary orthogonality regulariser.

    Forward is a plain :class:`nn.Linear` from ``in_features`` to ``out_dim``;
    :meth:`loss` returns ``MSE(W @ Wᵀ, I)`` (or ``Wᵀ @ W`` if the projection
    is over-complete) which the host model adds to its training objective
    via :func:`aux_loss`.
    """

    def __init__(self, in_features: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim, bias=bias)
        self.in_dim = in_features
        self.out_dim = out_dim
        identity_size = min(in_features, out_dim)
        self.register_buffer(
            "identity_matrix", torch.eye(identity_size), persistent=False
        )

    def loss(self) -> Tensor:
        weight = self.linear.weight
        gram = (
            weight @ weight.t() if self.in_dim > self.out_dim else weight.t() @ weight
        )
        target = self.identity_matrix.to(device=gram.device, dtype=gram.dtype)
        return F.mse_loss(gram, target)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.linear(input_tensor)


def _make_scaling_factors(channels: int) -> Tensor:
    return torch.cat(
        (
            torch.full((1, 1, channels), 0.5),
            torch.full((1, 1, channels), 0.5),
            torch.full((1, 1, channels), 0.5),
            torch.zeros((1, 1, channels)),
        ),
        dim=2,
    )


class WLS(nn.Module):
    r"""Wavelet Linear Scaling (analysis) block from Li et al., ICLR 2025.

    Auxiliary downsampling block: applies a 2D discrete wavelet transform,
    learnable per-subband scaling factors, and an :class:`OLP` channel
    mixer. Used as a building block inside the AuxT_enc side branch
    (:func:`build_wls_branch`) but also valid standalone.

    :class:`compressai.layers.wave.DWT2D` is imported lazily so this module
    stays importable without the ``pytorch_wavelets`` extra; constructing
    a :class:`WLS` instance triggers the dependency check.
    """

    def __init__(self, in_dim: int, out_dim: int, wave: str = "haar") -> None:
        super().__init__()
        from compressai.layers.wave import DWT2D

        self.dwt = DWT2D(wave=wave)
        self.olp = OLP(in_dim * 4, out_dim)
        self.scaling_factors = nn.Parameter(_make_scaling_factors(in_dim))

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.dwt(input_tensor)
        batch_size, _, height, width = output.shape
        output = output.view(batch_size, -1, height * width).permute(0, 2, 1)
        output = output * torch.exp(self.scaling_factors)
        output = self.olp(output)
        output = output.view(batch_size, height, width, -1)
        return output.permute(0, 3, 1, 2).contiguous()


class iWLS(nn.Module):
    r"""Inverse Wavelet Linear Scaling (synthesis) block from Li et al.,
    ICLR 2025.

    Mirror of :class:`WLS`: applies an :class:`OLP` channel mixer, undoes
    the learnable per-subband scaling, and reconstructs the spatial signal
    with the inverse 2D DWT.
    """

    def __init__(self, in_dim: int, out_dim: int, wave: str = "haar") -> None:
        super().__init__()
        from compressai.layers.wave import IDWT2D

        self.idwt = IDWT2D(wave=wave)
        self.olp = OLP(in_dim, out_dim * 4)
        self.scaling_factors = nn.Parameter(_make_scaling_factors(out_dim))

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.shape
        output = input_tensor.view(batch_size, -1, height * width).permute(0, 2, 1)
        output = self.olp(output)
        output = output / torch.exp(self.scaling_factors)
        output = output.view(batch_size, height, width, -1)
        output = output.permute(0, 3, 1, 2).contiguous()
        return self.idwt(output)


# ---------------------------------------------------------------------------
# Side-branch builders + walker (TCM-style integration)
# ---------------------------------------------------------------------------


def build_wls_branch(N: int, M: int) -> nn.ModuleList:
    """Standard 4-layer ``AuxT_enc`` analysis branch.

    Channel layout (matches Li et al., ICLR 2025 Sec. 3.2 reference impl):

    - ``WLS(3, 2N)`` — RGB image -> AuxT working width
    - ``WLS(2N, 2N)`` x 2 — interior stages
    - ``WLS(2N, M)`` — final stage matches the host's latent channels so
      the output can be summed into the last ``g_a`` layer.
    """
    return nn.ModuleList(
        [
            WLS(3, 2 * N),
            WLS(2 * N, 2 * N),
            WLS(2 * N, 2 * N),
            WLS(2 * N, M),
        ]
    )


def build_iwls_branch(N: int, M: int) -> nn.ModuleList:
    """Standard 4-layer ``AuxT_dec`` synthesis branch.

    Mirror of :func:`build_wls_branch`; final ``iWLS(2N, 3)`` reconstructs
    an RGB image.
    """
    return nn.ModuleList(
        [
            iWLS(M, 2 * N),
            iWLS(2 * N, 2 * N),
            iWLS(2 * N, 2 * N),
            iWLS(2 * N, 3),
        ]
    )


def forward_with_auxt(
    transform: nn.Sequential,
    auxiliary_layers: Optional[nn.ModuleList],
    merge_positions: Sequence[int],
    input_tensor: Tensor,
) -> Tensor:
    """Walk ``transform`` layer-by-layer, summing each AuxT[i] output at the
    matching ``merge_positions``.

    When ``auxiliary_layers is None`` (i.e. AuxT was not constructed) this
    collapses to ``transform(input_tensor)``, so hosts can call this
    unconditionally regardless of ``use_auxt``.

    Raises :class:`RuntimeError` when ``len(auxiliary_layers)`` does not
    match the number of merge positions actually consumed during the walk
    — usually a sign that ``merge_positions`` was computed against the
    wrong stage config.
    """
    if auxiliary_layers is None:
        return transform(input_tensor)

    if len(merge_positions) < len(auxiliary_layers):
        raise RuntimeError(
            "AuxT merge positions do not match auxiliary depth "
            f"(merge_positions has {len(merge_positions)} entries; "
            f"auxiliary_layers has {len(auxiliary_layers)})."
        )

    output = input_tensor
    auxiliary = input_tensor
    aux_index = 0
    for layer_index, layer in enumerate(transform):
        output = layer(output)
        if (
            aux_index < len(auxiliary_layers)
            and layer_index == merge_positions[aux_index]
        ):
            auxiliary = auxiliary_layers[aux_index](auxiliary)
            output = output + auxiliary
            aux_index += 1
    if aux_index != len(auxiliary_layers):
        raise RuntimeError(
            "AuxT merge positions do not match auxiliary depth "
            f"(merged {aux_index} of {len(auxiliary_layers)})."
        )
    return output


def compute_analysis_aux_positions(
    config: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Layer indices in ``g_a`` where ``AuxT_enc[i]`` outputs are summed in,
    for hosts using TCM's six-stage ``config`` convention.

    Derives the four boundaries by accumulating the depth of each
    :func:`compressai.models.tcm._make_mixed_stage` plus the inserted stride
    convolution. With the default TCM ``config = (2, 2, 2, 2, 2, 2)`` the
    positions land at ``(0, 3, 6, 9)`` of the 10-element ``g_a``
    Sequential.
    """
    return (
        0,
        config[0] + 1,
        config[0] + config[1] + 2,
        config[0] + config[1] + config[2] + 3,
    )


def compute_synthesis_aux_positions(
    config: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Mirror of :func:`compute_analysis_aux_positions` for ``g_s``.

    Uses ``config[3:]`` (the synthesis stages) so the positions land at
    ``(c3, c3+c4+1, c3+c4+c5+2, c3+c4+c5+3)``; with the default config
    this is ``(2, 5, 8, 9)``.
    """
    return (
        config[3],
        config[3] + config[4] + 1,
        config[3] + config[4] + config[5] + 2,
        config[3] + config[4] + config[5] + 3,
    )


# ---------------------------------------------------------------------------
# OLP regulariser aggregation (works for TCM-style and SAAF-style hosts)
# ---------------------------------------------------------------------------


def aux_loss(model: nn.Module) -> Tensor:
    """Sum :meth:`OLP.loss` over every :class:`OLP` in ``model``'s submodule
    tree, returning a 0-d :class:`Tensor`.

    Returns zero on the same device / dtype as the first model parameter
    when no :class:`OLP` modules are present, so callers can
    unconditionally add the result to their training objective regardless
    of whether AuxT is enabled.
    """
    losses = [module.loss() for module in model.modules() if isinstance(module, OLP)]
    if losses:
        return torch.stack(losses).sum()
    parameter = next(model.parameters())
    return torch.zeros((), device=parameter.device, dtype=parameter.dtype)


# ---------------------------------------------------------------------------
# State-dict helpers — checkpoint detection and upstream key normalization
# ---------------------------------------------------------------------------


def has_auxt_state(state_dict: Dict[str, Tensor]) -> bool:
    """``True`` when the state-dict carries any ``AuxT_enc.*`` /
    ``AuxT_dec.*`` keys.

    Hosts with an opt-in ``use_auxt`` parameter call this from
    :meth:`from_state_dict` to auto-detect whether the checkpoint
    requires AuxT branches.
    """
    return any(
        key.startswith("AuxT_enc.") or key.startswith("AuxT_dec.") for key in state_dict
    )


def is_auxt_wavelet_buffer_key(key: str) -> bool:
    """Match ``pytorch_wavelets``' own DWT/IDWT kernel buffer paths
    (``AuxT_enc.{k}.dwt.transform.h*`` / ``AuxT_dec.{k}.idwt.inverse.*``).

    Hosts that allow strict ``load_state_dict`` should add these to the
    "allowed missing" set: ``pytorch_wavelets`` re-registers the kernels
    at module construction time, so they are present in the model's own
    state-dict but may be absent from a checkpoint saved by a version
    that did not persist them.
    """
    if not (key.startswith("AuxT_enc.") or key.startswith("AuxT_dec.")):
        return False
    return ".dwt.transform." in key or ".idwt.inverse." in key


def is_auxt_upstream_wavelet_buffer_key(key: str) -> bool:
    """Match the wavelet kernel buffer names used by the upstream LIC_TCM
    AuxT release (``w_ll`` / ``w_lh`` / ``w_hl`` / ``w_hh`` for DWT and
    ``filters`` for IDWT).

    Convert scripts should drop these — the
    :mod:`pytorch_wavelets`-backed :class:`compressai.layers.wave.DWT2D`
    / :class:`IDWT2D` regenerate equivalent kernels at construction.
    """
    if key.startswith("AuxT_enc.") and ".dwt." in key:
        return key.rsplit(".", 1)[-1] in {"w_ll", "w_lh", "w_hl", "w_hh"}
    if key.startswith("AuxT_dec.") and ".idwt." in key:
        return key.rsplit(".", 1)[-1] == "filters"
    return False


def normalize_upstream_auxt_key(key: str) -> Optional[str]:
    """Translate an upstream PascalCase ``.OLP.`` attribute path to the
    compressai-canonical ``.olp.`` form, leaving non-AuxT keys alone.

    Returns ``None`` if the key is not an ``AuxT_enc.*`` / ``AuxT_dec.*``
    key (so callers can use a single ``if normalized := ...`` check).
    Pair with :func:`is_auxt_upstream_wavelet_buffer_key` to drop the
    upstream-specific DWT/IDWT kernel buffers in the same convert pass.
    """
    if not key.startswith(("AuxT_enc.", "AuxT_dec.")):
        return None
    return key.replace(".OLP.", ".olp.")
