# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-sample attention masking for packed sequences (Gemma 3, transformers >= 4.53).

WHY. A packed row holds several independent translation pairs back to back. Without masking, tokens of pair B
attend to pair A (different sentence, different language) and the loss of B depends on what happened to be packed
next to it. Position ids must restart too, otherwise RoPE sees B at positions it never has at inference.

WHAT GEMMA 3 NEEDS (read from transformers 4.57 `masking_utils.py` / `modeling_gemma3.py`):

  * Gemma3TextModel builds TWO masks per forward: `full_attention` (causal) for the global layers and
    `sliding_attention` (causal AND `q - kv < sliding_window`) for the local layers (5 local : 1 global in the
    270M model, window 512). A packed mask must be applied to BOTH. If a model call receives a dict
    {"full_attention": m, "sliding_attention": m_sw} as `attention_mask`, it is used as is (that is how
    `generate()` passes prepared masks) -> our "explicit" mode.
  * If `attention_mask is None` and `position_ids` is given and there is no cache, transformers itself derives
    the sample boundaries from `position_ids` (a boundary where position != previous + 1, see
    `find_packed_sequence_indices`) and ANDs them into both masks -> our "position_ids" mode (needs torch >= 2.6).
  * FlashAttention-2: `_flash_attention_forward` goes through `flash_attn_varlen_func` when it is handed
    `cu_seq_lens_q/k` + `max_length_q/k`, or (batch size 1 only!) when `position_ids` is non-monotonic
    (`_is_packed_sequence`). With batch size > 1 the position-id detection is SILENTLY OFF, so the padding-free
    path here always flattens the batch to [1, N] and passes the boundaries explicitly. The sliding window of the
    local layers is applied per sequence inside the varlen kernel (`window_size = (w - 1, w - 1)`).

Because the window is a *relative* distance and samples are contiguous, "causal AND same sample AND within
window" gives each token exactly the keys it would see if its sample were run alone. The test-suite proves this
numerically on a tiny random Gemma3 with a window smaller than the sample (tests/test_attention_equivalence.py).

Modes
    "explicit"      sdpa / eager : we build the two 4D masks from `segment_ids` (bool for sdpa, additive for eager)
    "position_ids"  sdpa / eager : we pass position_ids and let transformers derive the packed masks
    "flash"         flash_attention_2 or varlen : flattened batch + cu_seqlens (use `PackedCollator(mode="flatten")`)
    "none"          NEGATIVE CONTROL for tests: plain causal over the whole row (cross-sample leakage)

attn_implementation="varlen" (D-35) is our own attention function registered with transformers' `AttentionInterface`
(`varlen_attention_forward` below). It consumes exactly the kwargs the "flash" mode produces (cu_seq_lens_q/k,
max_length_q/k), calls `torch.nn.attention.varlen.varlen_attn` (the FlashAttention kernel shipped inside torch, no
`flash-attn` package) with `window_size=(sliding_window - 1, 0)` on the local layers, and needs no [T, T] mask at all.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

MASK_MODES = ("explicit", "position_ids", "flash", "none")
VARLEN = "varlen"                                  # attn_implementation name of the torch-varlen attention (this module registers it)
FLASH_IMPLS = ("flash_attention_2", VARLEN)        # the implementations driven by mask_mode="flash" (flattened batch + cu_seqlens)


# ------------------------------------------------------------------------------------------------ numpy side
def segments_from_position_ids(position_ids: np.ndarray) -> np.ndarray:
    """Segment index (0, 1, 2, ...) per token from position ids that restart at every sample.

    Same rule as transformers' `find_packed_sequence_indices`: a new segment starts wherever the position is not
    previous + 1. Works for [T] and [B, T]."""
    pos = np.asarray(position_ids)
    squeeze = pos.ndim == 1
    if squeeze:
        pos = pos[None]
    prev = np.concatenate([pos[:, :1] - 1, pos[:, :-1]], axis=1)
    seg = np.cumsum(pos != prev + 1, axis=1)
    return seg[0] if squeeze else seg


def block_diagonal_causal_mask_np(segment_ids: np.ndarray, sliding_window: Optional[int] = None) -> np.ndarray:
    """Boolean [B, 1, T, T], True = may attend. causal AND same segment (AND q - kv < window when given)."""
    seg = np.asarray(segment_ids)
    if seg.ndim == 1:
        seg = seg[None]
    B, T = seg.shape
    q = np.arange(T)[:, None]
    k = np.arange(T)[None, :]
    causal = k <= q
    if sliding_window is not None:
        causal = causal & (q - k < sliding_window)
    same = seg[:, :, None] == seg[:, None, :]
    return (same & causal[None])[:, None]


# ------------------------------------------------------------------------------------------------ torch side
def block_diagonal_causal_mask(segment_ids, sliding_window: Optional[int] = None):
    """torch version of `block_diagonal_causal_mask_np`; bool [B, 1, T, T] on the device of `segment_ids`."""
    import torch

    seg = segment_ids
    if seg.dim() == 1:
        seg = seg.unsqueeze(0)
    T = seg.shape[1]
    idx = torch.arange(T, device=seg.device)
    causal = idx[None, :] <= idx[:, None]
    if sliding_window is not None:
        causal = causal & ((idx[:, None] - idx[None, :]) < sliding_window)
    same = seg[:, :, None] == seg[:, None, :]
    return (same & causal[None]).unsqueeze(1)


def additive_mask(mask_bool, dtype):
    """bool (True = attend) -> additive float mask (0 / finfo.min), what the eager attention adds to the scores."""
    import torch

    out = torch.zeros(mask_bool.shape, dtype=dtype, device=mask_bool.device)
    return out.masked_fill(~mask_bool, torch.finfo(dtype).min)


def packed_masks_for_gemma3(segment_ids, sliding_window: Optional[int], attn_implementation: str, dtype=None):
    """The `attention_mask` dict Gemma3TextModel accepts: {"full_attention": ..., "sliding_attention": ...}."""
    import torch

    full = block_diagonal_causal_mask(segment_ids, None)
    local = block_diagonal_causal_mask(segment_ids, sliding_window) if sliding_window else full
    if attn_implementation == "eager":
        dt = dtype or torch.float32
        full, local = additive_mask(full, dt), additive_mask(local, dt)
    return {"full_attention": full, "sliding_attention": local}


def model_inputs(batch: Dict[str, Any], *, attn_implementation: str, mask_mode: str, sliding_window: Optional[int],
                 dtype=None, kernel_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Keyword arguments for `Gemma3ForCausalLM.forward` / `.model.forward` that give a packed batch its
    block-diagonal attention. `batch` holds torch tensors from `PackedCollator` (already on the device).

    The returned dict never contains `labels`; the loss is computed by csw.vendor.gemma_mt.losses.

    `kernel_options` (flex_attention only): the Triton tile options transformers forwards to torch's flex_attention (e.g.
    {"BLOCK_M": 32, "BLOCK_N": 32, ...}); head_dim 256 needs small tiles on sm_120 (see DECISIONS D-35)."""
    kw = _model_inputs(batch, attn_implementation=attn_implementation, mask_mode=mask_mode, sliding_window=sliding_window,
                       dtype=dtype)
    if kernel_options and attn_implementation == "flex_attention":
        kw["kernel_options"] = dict(kernel_options)
    return kw


def _model_inputs(batch: Dict[str, Any], *, attn_implementation: str, mask_mode: str, sliding_window: Optional[int],
                  dtype=None) -> Dict[str, Any]:
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}")
    ids = batch["input_ids"]
    # use_cache=False is REQUIRED: with a cache object present transformers does not derive packed masks from
    # position_ids (found by experiment on 4.57: errors of ~0.8 in logits with the cache on, 1e-7 with it off).
    if mask_mode == "none":
        return {"input_ids": ids, "attention_mask": None, "position_ids": None, "use_cache": False}
    pos = batch["position_ids"]
    if mask_mode == "flash":
        if attn_implementation not in FLASH_IMPLS:
            raise ValueError(f"mask_mode='flash' requires attn_implementation in {FLASH_IMPLS}")
        if ids.shape[0] != 1:
            raise ValueError("flash varlen path needs a flattened batch of shape [1, N] (PackedCollator(mode='flatten'))")
        cu = batch["cu_seqlens"].to(device=ids.device, dtype=_int32())
        # varlen: the model must not build any mask (a dict with None entries is used as is by Gemma3TextModel and reaches
        # every attention layer as attention_mask=None; the kernel gets its structure from cu_seq_lens_q/k alone)
        no_mask = {"full_attention": None, "sliding_attention": None} if attn_implementation == VARLEN else None
        return {"input_ids": ids, "attention_mask": no_mask, "position_ids": pos, "cu_seq_lens_q": cu,
                "cu_seq_lens_k": cu, "max_length_q": int(batch["max_seqlen"]), "max_length_k": int(batch["max_seqlen"]),
                "use_cache": False}
    if attn_implementation in FLASH_IMPLS:
        raise ValueError(f"use mask_mode='flash' with {attn_implementation}")
    if mask_mode == "position_ids":
        return {"input_ids": ids, "attention_mask": None, "position_ids": pos, "use_cache": False}
    masks = packed_masks_for_gemma3(batch["segment_ids"], sliding_window, attn_implementation, dtype)
    return {"input_ids": ids, "attention_mask": masks, "position_ids": pos, "use_cache": False}


def _int32():
    import torch

    return torch.int32


def default_mask_mode(attn_implementation: str) -> str:
    """flash / varlen: the flattened stream; flex_attention: transformers builds its BlockMask from position_ids (explicit 4-D
    boolean masks are NOT understood by its flex integration: found by experiment, the loss was wrong from step 1); else explicit."""
    if attn_implementation in FLASH_IMPLS:
        return "flash"
    return "position_ids" if attn_implementation == "flex_attention" else "explicit"


def collator_mode_for(attn_implementation: str) -> str:
    """Which `PackedCollator` layout goes with an attention implementation."""
    return "flatten" if attn_implementation in FLASH_IMPLS else "padded"


# ------------------------------------------------------------------------------------------------ varlen attention (D-35)
def _reference_attention(q, k, v, cu, scaling, sliding_window):
    """Pure-torch reference of what the varlen kernel computes, used on CPU and for fp32 (the FlashAttention kernel has
    no fp32): q [T, H, D], k / v [T, Hkv, D] -> [T, H, D]. One SDPA call per sample with its own causal (+ window) mask,
    so memory is O(longest sample ^ 2), not O(T ^ 2); a Python loop, hence slow: for tests and fp32 debugging, not for
    bf16 training (that uses the kernel)."""
    import torch
    import torch.nn.functional as F

    T, H, D = q.shape
    rep = H // k.shape[1]
    bounds = [int(x) for x in cu.tolist()]
    out = torch.zeros_like(q)
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        L = s1 - s0
        if L <= 0:
            continue
        idx = torch.arange(L, device=q.device)
        allowed = idx[None, :] <= idx[:, None]
        if sliding_window:
            allowed = allowed & ((idx[:, None] - idx[None, :]) < sliding_window)
        qh = q[s0:s1].transpose(0, 1)[None]                                       # [1, H, L, D]
        kh = k[s0:s1].transpose(0, 1).repeat_interleave(rep, dim=0)[None]
        vh = v[s0:s1].transpose(0, 1).repeat_interleave(rep, dim=0)[None]
        out[s0:s1] = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=allowed[None, None], scale=scaling)[0].transpose(0, 1)
    return out


def _kernel_dtype(x):
    """dtype the FlashAttention kernel must run in: the autocast dtype when autocast is on (this custom function is not on
    torch's autocast list, unlike SDPA, so it has to do the cast itself), else the tensor's own dtype."""
    import torch

    dev = x.device.type
    if dev in ("cuda", "cpu") and torch.is_autocast_enabled(dev):
        return torch.get_autocast_dtype(dev)
    return x.dtype


def varlen_attention_forward(module, query, key, value, attention_mask=None, dropout: float = 0.0, scaling=None,
                             sliding_window=None, softcap=None, **kwargs):
    """transformers `AttentionInterface` function: causal (+ sliding window) attention inside every sample of a flattened
    [1, N] stream whose boundaries are `cu_seq_lens_q` / `max_length_q` (the kwargs of the "flash" mask mode).

    Layouts (transformers 4.57 / 5.19, Gemma3Attention): query [1, H, N, D], key / value [1, Hkv, N, D] (NOT repeated
    for GQA), `scaling` = query_pre_attn_scalar ** -0.5, `sliding_window` = window of this layer or None (global).
    Returns (attn_output [1, N, H, D], None).

    Semantics are those of the explicit-mask path: token t of a sample attends to the tokens s of the SAME sample with
    s <= t and (global layer) no further limit, (local layer) t - s < sliding_window  <=>  window_size=(w - 1, 0)."""
    import torch

    cu = kwargs.get("cu_seq_lens_q")
    max_q = kwargs.get("max_length_q")
    if cu is None or max_q is None:
        raise ValueError("attn_implementation='varlen' needs cu_seq_lens_q / max_length_q (mask_mode='flash' with a flattened "
                         "PackedCollator batch); a padded 2-D attention mask is not supported (e.g. the RL policy forward)")
    if dropout:
        raise ValueError("varlen attention has no dropout")
    if softcap is not None or getattr(module, "attn_logit_softcapping", None) is not None:
        raise ValueError("varlen attention does not support attention logit softcapping")
    if query.shape[0] != 1:
        raise ValueError(f"varlen attention needs a flattened batch of shape [1, N], got batch {query.shape[0]}")
    if not getattr(module, "is_causal", True):
        raise ValueError("varlen attention is causal only")

    q = query[0].transpose(0, 1)                                         # [N, H, D] views
    k = key[0].transpose(0, 1)
    v = value[0].transpose(0, 1)
    dt = _kernel_dtype(q)
    use_kernel = q.is_cuda and dt in (torch.bfloat16, torch.float16)
    if not use_kernel:
        out = _reference_attention(q.to(dt), k.to(dt), v.to(dt), cu, scaling, sliding_window)
        return out.unsqueeze(0).contiguous(), None
    from torch.nn.attention.varlen import varlen_attn

    window = (int(sliding_window) - 1, 0) if sliding_window else (-1, 0)
    out = varlen_attn(q.to(dt).contiguous(), k.to(dt).contiguous(), v.to(dt).contiguous(), cu, cu, int(max_q), int(max_q),
                      scale=scaling, window_size=window, enable_gqa=(q.shape[1] != k.shape[1]))
    return out.unsqueeze(0), None


def register_varlen_attention() -> None:
    """Register `attn_implementation="varlen"` with transformers (idempotent). Must run before a model is built / loaded
    with that name; `modeling.load_model` and `modeling.build_tiny_model` do it."""
    from transformers import AttentionInterface

    AttentionInterface.register(VARLEN, varlen_attention_forward)
