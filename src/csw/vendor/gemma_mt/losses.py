# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token-level losses without ever materialising [tokens, 262144] logits for the whole batch.

Gemma's vocabulary is 262,144 rows. Logits for 16k tokens in fp32 are 17 GB, which is why the loss is computed
only at positions that carry a label and in chunks, each chunk wrapped in activation checkpointing so that the
backward pass recomputes (and immediately frees) its logits.

Conventions
  * `labels` has the SAME alignment as `input_ids` (HF convention). The prediction at position t is scored against
    labels[t + 1]; labels[:, 0] is never used.
  * losses are returned as SUMS plus the token count; normalisation (token-weighted over a whole accumulated
    batch) is done by the caller.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def _chunk_logp(h: torch.Tensor, w: torch.Tensor, t: torch.Tensor, softcap: Optional[float], temperature: float):
    logits = F.linear(h, w).float()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    if temperature != 1.0:
        logits = logits / temperature
    return logits.gather(-1, t.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


LOSS_IMPLS = ("chunked", "fused")


def token_logprobs_chunked(hidden: torch.Tensor, lm_head_weight: torch.Tensor, targets: torch.Tensor, *,
                           softcap: Optional[float] = None, temperature: float = 1.0, chunk: int = 512,
                           use_checkpoint: bool = True, impl: str = "chunked") -> torch.Tensor:
    """log p(targets | hidden) in fp32 for hidden [N, H], targets [N]. Differentiable; peak extra memory is one
    chunk of logits.

    impl="chunked" (default): the reference implementation below. impl="fused": `fused_loss.fused_linear_logp` (Triton
    log-sum-exp, fp32 weight-gradient accumulation; D-36); final-logit softcapping is not implemented there, so a model
    with softcapping silently keeps the reference path."""
    if hidden.shape[0] != targets.shape[0]:
        raise ValueError(f"hidden {tuple(hidden.shape)} vs targets {tuple(targets.shape)}")
    if impl not in LOSS_IMPLS:
        raise ValueError(f"loss_impl must be one of {LOSS_IMPLS}, got {impl!r}")
    if impl == "fused" and softcap is None:
        from csw.vendor.gemma_mt.fused_loss import fused_linear_logp

        return fused_linear_logp(hidden, lm_head_weight, targets, chunk=chunk, temperature=temperature)
    outs = []
    for s in range(0, hidden.shape[0], chunk):
        h, t = hidden[s: s + chunk], targets[s: s + chunk]
        if use_checkpoint and torch.is_grad_enabled() and (h.requires_grad or lm_head_weight.requires_grad):
            outs.append(checkpoint(_chunk_logp, h, lm_head_weight, t, softcap, temperature, use_reentrant=False))
        else:
            outs.append(_chunk_logp(h, lm_head_weight, t, softcap, temperature))
    if not outs:
        return hidden.new_zeros((0,), dtype=torch.float32)
    return torch.cat(outs)


def final_hidden(model: Any, **model_kwargs) -> torch.Tensor:
    """Last hidden state (after the final norm) of a Gemma3ForCausalLM; the LM head is applied by the caller."""
    inner = getattr(model, "module", model)
    return inner.model(**model_kwargs).last_hidden_state


def lm_head_weight_and_softcap(model: Any) -> Tuple[torch.Tensor, Optional[float]]:
    inner = getattr(model, "module", model)
    return inner.lm_head.weight, getattr(inner.config, "final_logit_softcapping", None)


def sft_selected_logp(model: Any, model_kwargs: dict, labels: torch.Tensor, chunk: int = 512, loss_impl: str = "chunked"):
    """(logp [n] for every labelled position in row-major order, sel [B, T-1] bool mask of those positions)."""
    h = final_hidden(model, **model_kwargs)
    tgt = labels[:, 1:]
    sel = tgt != IGNORE_INDEX
    if int(sel.sum().item()) == 0:
        return h.sum().reshape(1)[:0] * 0.0, sel
    w, cap = lm_head_weight_and_softcap(model)
    return token_logprobs_chunked(h[:, :-1][sel], w, tgt[sel], softcap=cap, chunk=chunk, impl=loss_impl), sel


def sft_nll_sum(model: Any, model_kwargs: dict, labels: torch.Tensor, chunk: int = 512,
                loss_impl: str = "chunked") -> Tuple[torch.Tensor, int]:
    """Sum of -log p over the labelled positions of the batch and their count."""
    logp, sel = sft_selected_logp(model, model_kwargs, labels, chunk, loss_impl)
    n = int(sel.sum().item())
    if n == 0:
        return logp.sum(), 0
    return -(logp.sum()), n


def completion_logprobs(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_width: int,
                        completion_mask: torch.Tensor, temperature: float = 1.0, chunk: int = 512,
                        loss_impl: str = "chunked") -> torch.Tensor:
    """Per-token log-probs of the completion part of left-padded prompt + right-padded completion sequences.

    input_ids / attention_mask: [B, P + C]  (prompt left-padded to width P, completion right-padded to C)
    completion_mask: [B, C]  1 on real completion tokens (the stop token included), 0 on padding
    returns [B, C] fp32, exactly 0 where completion_mask == 0.

    Index alignment: completion token j sits at absolute index P + j and is predicted by the hidden state at index
    P + j - 1; the logits are scored under `logits / temperature` (the distribution the sampler draws from)."""
    B, total = input_ids.shape
    C = total - prompt_width
    if completion_mask.shape != (B, C):
        raise ValueError(f"completion_mask {tuple(completion_mask.shape)} != {(B, C)}")
    pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    h = final_hidden(model, input_ids=input_ids, attention_mask=attention_mask, position_ids=pos, use_cache=False)
    h_pred = h[:, prompt_width - 1: total - 1]                       # [B, C, H]
    tgt = input_ids[:, prompt_width:]                                # [B, C]
    sel = completion_mask.bool()
    out = torch.zeros(B, C, dtype=torch.float32, device=input_ids.device)
    if sel.any():
        w, cap = lm_head_weight_and_softcap(model)
        lp = token_logprobs_chunked(h_pred[sel], w, tgt[sel], softcap=cap, temperature=temperature, chunk=chunk,
                                    impl=loss_impl)
        out = out.masked_scatter(sel, lp)
    return out
