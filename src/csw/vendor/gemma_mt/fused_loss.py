# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fused LM-head log-probability: log p(target | hidden) without any fp32 [rows, 262144] tensor (D-36).

The reference implementation (`losses.token_logprobs_chunked`, `loss_impl: chunked`) materialises, per chunk of rows,
bf16 logits -> a fp32 copy -> (exp / sum / max passes of torch.logsumexp) and, in the backward pass (the chunk is
re-computed under activation checkpointing), the same again plus the fp32 softmax. At a vocabulary of 262,144 those
element-wise passes are memory bound and were ~30 % of a training step (CUDA profile, RTX 5070).

This module keeps what is numerically relevant and drops the traffic:

  forward   logits = h @ w^T  in the autocast dtype (bf16, exactly the GEMM the reference runs), then ONE Triton pass over
            every row that reads the bf16 logits and accumulates the log-sum-exp in fp32 (online max / sum); the target
            logit is gathered. Nothing of size [rows, V] is kept: only lse [rows] (fp32).
  backward  the logits are recomputed chunk by chunk (same GEMM), turned IN PLACE into d(logp)/d(logits) * g =
            g * (onehot - softmax) by one Triton pass (softmax in fp32, stored back in bf16, which is also what the
            reference feeds to its backward GEMMs), then dh = dlogits @ w and dw += dlogits^T @ h with the weight gradient
            ACCUMULATED IN FP32 across chunks (the reference accumulates the per-chunk bf16 weight gradients in bf16).

Everything is one autograd Function over all rows, so the activation-checkpoint wrapper is not needed. Rows are processed
`chunk` at a time (peak extra memory: one bf16 [chunk, V] tile + the fp32 weight-gradient buffer).

CPU and non-Triton fallback: the same Function with plain torch ops instead of the two kernels, so the algebra (and the
tests that run on the laptop) does not depend on a GPU. Not supported here (the caller falls back to the reference
implementation): final-logit softcapping. Temperature is applied by scaling the hidden states."""

from __future__ import annotations

from typing import Optional

import torch

try:                                           # Triton ships with the CUDA builds of torch
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:                              # pragma: no cover - CPU-only machines
    triton = tl = None
    _HAVE_TRITON = False

_BLOCK = 4096


if _HAVE_TRITON:

    @triton.jit
    def _lse_kernel(X, LSE, V, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        base = X + row * stride
        m = tl.full([BLOCK], float("-inf"), tl.float32)
        s = tl.zeros([BLOCK], tl.float32)
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            ok = cols < V
            x = tl.load(base + cols, mask=ok, other=float("-inf")).to(tl.float32)
            m_new = tl.maximum(m, x)
            alpha = tl.where(m == m_new, 1.0, tl.exp(m - m_new))
            p = tl.where(ok, tl.exp(x - m_new), 0.0)
            s = s * alpha + p
            m = m_new
        big = tl.max(m, axis=0)
        total = tl.sum(s * tl.exp(m - big), axis=0)
        tl.store(LSE + row, big + tl.log(total))

    @triton.jit
    def _dlogits_kernel(X, LSE, TGT, G, V, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = cols < V
        ptr = X + row * stride + cols
        x = tl.load(ptr, mask=ok, other=float("-inf")).to(tl.float32)
        lse = tl.load(LSE + row)
        g = tl.load(G + row)
        t = tl.load(TGT + row)
        p = tl.exp(x - lse)
        d = g * (tl.where(cols == t, 1.0, 0.0) - p)
        tl.store(ptr, d.to(X.dtype.element_ty), mask=ok)


def _use_triton(x: torch.Tensor) -> bool:
    return _HAVE_TRITON and x.is_cuda


def _lse_rows(logits: torch.Tensor) -> torch.Tensor:
    """log-sum-exp over the last dim of a contiguous [n, V] tensor, fp32 result [n]."""
    n, V = logits.shape
    if _use_triton(logits) and n:
        out = torch.empty(n, dtype=torch.float32, device=logits.device)
        _lse_kernel[(n,)](logits, out, V, logits.stride(0), BLOCK=_BLOCK, num_warps=8)
        return out
    return torch.logsumexp(logits.float(), dim=-1)


def _dlogits_(logits: torch.Tensor, lse: torch.Tensor, tgt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """In place: logits <- g * (onehot(tgt) - softmax(logits)), softmax from the fp32 `lse`; returns `logits`."""
    n, V = logits.shape
    if _use_triton(logits) and n:
        _dlogits_kernel[(n, triton.cdiv(V, _BLOCK))](logits, lse, tgt, g, V, logits.stride(0), BLOCK=_BLOCK, num_warps=8)
        return logits
    p = torch.exp(logits.float() - lse[:, None])
    p.neg_()
    p.scatter_add_(1, tgt[:, None], torch.ones_like(p[:, :1]))
    p.mul_(g[:, None])
    logits.copy_(p.to(logits.dtype))
    return logits


def _accumulate_mm_(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    """acc (fp32) += a @ b, with a reduced-precision GEMM whose result is fp32 (no bf16 rounding of the running sum and no
    temporary the size of acc: torch >= 2.8 `addmm(..., out_dtype=)`); falls back to a bf16 product added in fp32."""
    if a.dtype != torch.float32 and acc.dtype == torch.float32:
        try:
            torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
            return
        except torch.OutOfMemoryError:
            raise
        except (TypeError, RuntimeError):            # torch without addmm(out_dtype=) or a device that lacks it
            pass
    acc += (a @ b).to(acc.dtype)


def _compute_dtype(h: torch.Tensor) -> torch.dtype:
    dev = h.device.type
    if dev in ("cuda", "cpu") and torch.is_autocast_enabled(dev):
        return torch.get_autocast_dtype(dev)
    return h.dtype


class _FusedLinearLogp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, t, chunk):
        cd = _compute_dtype(h)
        hc, wc = h.to(cd), w.to(cd)
        N = h.shape[0]
        lse = torch.empty(N, dtype=torch.float32, device=h.device)
        tl_ = torch.empty(N, dtype=torch.float32, device=h.device)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            logits = (hc[s:e] @ wc.T).contiguous()
            lse[s:e] = _lse_rows(logits)
            tl_[s:e] = logits.gather(1, t[s:e, None]).squeeze(1).float()
            del logits
        ctx.save_for_backward(h, w, t, lse)
        ctx.cd, ctx.chunk = cd, chunk
        return tl_ - lse

    @staticmethod
    def backward(ctx, g):
        h, w, t, lse = ctx.saved_tensors
        cd, chunk = ctx.cd, ctx.chunk
        hc, wc = h.to(cd), w.to(cd)
        g = g.contiguous().float()
        N = h.shape[0]
        dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
        dw = torch.zeros(w.shape, dtype=torch.float32, device=w.device) if ctx.needs_input_grad[1] else None
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = _dlogits_((hc[s:e] @ wc.T).contiguous(), lse[s:e], t[s:e], g[s:e])
            if dh is not None:
                dh[s:e] = (d @ wc).to(dh.dtype)
            if dw is not None:
                _accumulate_mm_(dw, d.T, hc[s:e])
            del d
        return dh, (dw.to(w.dtype) if dw is not None else None), None, None


def fused_linear_logp(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, *, chunk: int = 2048,
                      temperature: float = 1.0) -> torch.Tensor:
    """log softmax(hidden @ weight^T / temperature)[targets], fp32 [N]; differentiable in hidden and weight.
    hidden [N, H], weight [V, H], targets [N] (int64). Under autocast the GEMMs run in the autocast dtype."""
    if hidden.shape[0] != targets.shape[0]:
        raise ValueError(f"hidden {tuple(hidden.shape)} vs targets {tuple(targets.shape)}")
    if hidden.shape[0] == 0:
        return hidden.new_zeros((0,), dtype=torch.float32)
    if temperature != 1.0:
        hidden = hidden / temperature
    return _FusedLinearLogp.apply(hidden.contiguous(), weight, targets.contiguous(), int(chunk))
