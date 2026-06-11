"""Dictionary-based channel-context heads for DCAE / SAAF.

DCAE (Lu et al., CVPR 2025) and SAAF (Ma et al., CVPR 2026) share an
entropy stack that augments the per-slice channel support with a learned
**shared dictionary** (a single ``dt: nn.Parameter`` of shape
``(dict_num, dictionary_dim)`` that all K slices cross-attend against).
This module provides:

- :class:`SharedDictionary` — owns the ``dt`` Parameter at the model level
  (path: ``shared_dictionary.dt``). Heads access it via a closure stored as
  a plain Python attribute so the parameter is not duplicated under K
  per-slice paths in the state-dict.
- :class:`DictionaryMeanScaleContextHead` — per-slice channel-context head
  that runs :class:`MutiScaleDictionaryCrossAttentionGLU` on its input,
  concatenates the cross-attention output with the input, and feeds the
  combined ``support`` tensor into separate ``mean_cc`` / ``scale_cc``
  Sequentials. Drops into
  model-local side-parameter channel-groups paths, just like
  :class:`~compressai.models._helpers.channel_context.MeanScaleContextHead`.
- :func:`build_dictionary_mean_scale_head` — convenience factory for the
  DCAE / SAAF dictionary context heads.

Why a closure for the shared dictionary, not a submodule? Storing
``SharedDictionary`` as a child module of every head would either:
(a) duplicate the ``dt`` Parameter under K paths in :meth:`state_dict`
(verified experimentally — :meth:`nn.Module.state_dict` traverses each
referencing submodule independently), or (b) require an invasive change to
upstream :class:`ChannelGroupsLatentCodec` to add a ``shared_modules`` slot.
Storing as a plain Python attribute (Callable) sidesteps both.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers.attn.dictionary import MutiScaleDictionaryCrossAttentionGLU
from compressai.models._helpers.slice_helpers import make_entropy_transform

__all__ = [
    "DictionaryMeanScaleContextHead",
    "SharedDictionary",
    "build_dictionary_mean_scale_head",
]


class SharedDictionary(nn.Module):
    """Holds the learned dictionary tensor cross-attended by DCAE / SAAF heads.

    Owned by the model (e.g. ``self.shared_dictionary = SharedDictionary(...)``);
    a closure over this instance is threaded into every per-slice
    :class:`DictionaryMeanScaleContextHead` so all heads share the same
    underlying ``dt`` Parameter without duplicating it in the state-dict.
    """

    dt: nn.Parameter

    def __init__(self, dict_num: int, dictionary_dim: int) -> None:
        super().__init__()
        self.dt = nn.Parameter(torch.randn(dict_num, dictionary_dim))

    def expand_for(self, batch_size: int) -> Tensor:
        """Broadcast ``dt`` to ``(batch_size, dict_num, dictionary_dim)``.

        :class:`MutiScaleDictionaryCrossAttentionGLU` expects a per-batch
        dictionary tensor; we materialise a view here without copying.
        """
        return self.dt.unsqueeze(0).expand(batch_size, -1, -1)


class DictionaryMeanScaleContextHead(nn.Module):
    """Channel-context head with shared dictionary cross-attention.

    Forward flow (DCAE / SAAF, with
    model-local side-parameter channel-groups path)::

        x = cat([latent_means(M), latent_scales(M), *prev_y_hat], dim=1)
        dict_info = cross_attention(x, shared_dt)         # (B, dict_output_ch, H, W)
        support   = cat([x, dict_info], dim=1)            # (B, support_ch + dict_output_ch, H, W)
        mean      = mean_cc(support)                      # (B, slice_ch, H, W)
        scale     = scale_cc(support)
        out       = cat([scale, mean], dim=1)             # chunks=("scales","means")
        if emit_mean_support: out = cat([out, support], dim=1)

    The trailing ``support`` block (when ``emit_mean_support=True``) is
    consumed by the model-local LRP Gaussian leaf with
    ``mean_support_trail_channels = support_ch + dict_output_ch`` to recover
    the upstream ``cat(support, y_hat)`` LRP input layout, enabling
    byte-for-byte transfer of upstream LRP weights.

    Note on input ordering: the upstream DCAE source assembles its query as
    ``cat([latent_scales, latent_means, *support_slices])`` (scales before
    means), whereas the containerized wiring used here produces
    ``cat([latent_means, latent_scales, *prev_y_hat])`` (means before
    scales). The ``cc_mean`` / ``cc_scale`` /
    ``cross_attention`` / ``lrp_transform`` first-conv weights from upstream
    DCAE / SAAF checkpoints therefore need their leading 2M input channels
    swapped (channels ``[0:M]`` ↔ ``[M:2M]``) at conversion time —
    ``examples/convert_{dcae,saaf}_checkpoint.py`` handles this rename.
    """

    cross_attention: MutiScaleDictionaryCrossAttentionGLU
    mean_cc: nn.Module
    scale_cc: nn.Module

    def __init__(
        self,
        cross_attention: MutiScaleDictionaryCrossAttentionGLU,
        mean_cc: nn.Module,
        scale_cc: nn.Module,
        *,
        dictionary_provider: Callable[[int], Tensor],
        emit_mean_support: bool = False,
    ) -> None:
        super().__init__()
        self.cross_attention = cross_attention
        self.mean_cc = mean_cc
        self.scale_cc = scale_cc
        # Plain Python attribute (Callable) — not registered as a submodule.
        # See module docstring for the rationale.
        self._dictionary_provider = dictionary_provider
        self.emit_mean_support = bool(emit_mean_support)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        dictionary = self._dictionary_provider(batch_size)
        dict_info = self.cross_attention(x, dictionary)
        support = torch.cat([x, dict_info], dim=1)
        mean = self.mean_cc(support)
        scale = self.scale_cc(support)
        out = torch.cat([scale, mean], dim=1)
        if self.emit_mean_support:
            out = torch.cat([out, support], dim=1)
        return out


def build_dictionary_mean_scale_head(
    slice_ch: int,
    support_ch: int,
    *,
    shared_dictionary: SharedDictionary,
    dict_output_ch: int,
    cross_attention_kwargs: Optional[dict] = None,
    widths: Sequence[int] = (224, 128),
    emit_mean_support: bool = False,
) -> DictionaryMeanScaleContextHead:
    """Construct a :class:`DictionaryMeanScaleContextHead`.

    Parameters
    ----------
    slice_ch
        Output channel count of each ``mean_cc`` / ``scale_cc`` head
        (= channel width of the slice being predicted).
    support_ch
        Input channel count handed to the head by
        model-local side-parameter channel-groups path. This equals
        ``2 * M + slice_ch * support_count`` (the head does no internal
        split: the cross-attention treats it as a flat support tensor).
    shared_dictionary
        :class:`SharedDictionary` instance owned by the model. The head
        captures it via a closure to avoid duplicating ``dt`` in the
        state-dict (one path: ``shared_dictionary.dt``, regardless of K).
    dict_output_ch
        Output channel count of
        :class:`MutiScaleDictionaryCrossAttentionGLU`. DCAE / SAAF use
        ``M`` so the cross-attention contributes another M channels to the
        ``support`` tensor that ``mean_cc`` / ``scale_cc`` consume.
    cross_attention_kwargs
        Extra kwargs forwarded to
        :class:`MutiScaleDictionaryCrossAttentionGLU` (``head_num``,
        ``mlp_rate``, ``qkv_bias``). ``dictionary_dim`` is filled in
        automatically from ``shared_dictionary.dt.size(1)``.
    widths
        Hidden conv widths inside the ``mean_cc`` / ``scale_cc``
        Sequentials. Defaults to ``(224, 128)`` (the DCAE / SAAF /
        TCM / CCA convention).
    emit_mean_support
        When ``True``, append the ``support`` tensor to the head output.
        Pair with
        the model-local LRP Gaussian leaf and
        ``mean_support_trail_channels = support_ch + dict_output_ch`` to
        reproduce the upstream DCAE / SAAF LRP input layout
        ``cat(support, y_hat)``.
    """
    cross_attention_kwargs = dict(cross_attention_kwargs or {})
    cross_attention_kwargs.setdefault("dictionary_dim", shared_dictionary.dt.size(1))
    cross_attention = MutiScaleDictionaryCrossAttentionGLU(
        input_dim=support_ch,
        output_dim=dict_output_ch,
        **cross_attention_kwargs,
    )
    cc_in_ch = support_ch + dict_output_ch
    mean_cc = make_entropy_transform(cc_in_ch, slice_ch, widths=widths)
    scale_cc = make_entropy_transform(cc_in_ch, slice_ch, widths=widths)

    def dictionary_provider(batch_size: int) -> Tensor:
        return shared_dictionary.expand_for(batch_size)

    return DictionaryMeanScaleContextHead(
        cross_attention=cross_attention,
        mean_cc=mean_cc,
        scale_cc=scale_cc,
        dictionary_provider=dictionary_provider,
        emit_mean_support=emit_mean_support,
    )
