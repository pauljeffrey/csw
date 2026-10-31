# SPDX-License-Identifier: AGPL-3.0-or-later
"""GRPO building blocks (pure torch, no model): advantages, KL estimators, the token-level loss.

Notation (one prompt x, G sampled completions y_1..y_G, rewards r_1..r_G, per-token log-probs under the CURRENT
policy lp_{i,t}, under the frozen reference lp^ref_{i,t}, mask m_{i,t} = 1 on real completion tokens):

  advantage          A_i = (r_i - mean_j r_j) / (std_j r_j + eps)      [norm="mean_std", the repo's choice]
                     A_i =  r_i - mean_j r_j                           [norm="mean"]
                     (std is the population std over the group; a group whose std < flat_threshold has A = 0)
                     A_i is then multiplied by the per-language weight w_i in [0, 1] (reliability of QE there).
  policy gradient    L_pg(i,t) = - A_i * rho_{i,t}                 rho = exp(lp - lp_old)
                     with clipping:  - min(rho A, clip(rho, 1-eps, 1+eps_hi) A)
                     With ONE update per sampled batch lp_old = stop_grad(lp): rho == 1 and the gradient is exactly
                     - A_i * grad lp_{i,t}  (REINFORCE with a group baseline), nothing ever clips.
  KL penalty         d = lp^ref - lp
                     k3 = exp(d) - d - 1    >= 0, unbiased estimate of KL(pi || ref) under samples from pi
                         (its GRADIENT, with samples held fixed, is the gradient of KL(ref || pi): this is what
                          GRPO/DeepSeekMath does and what the repo does)
                     k2 = d^2 / 2           its gradient IS the gradient of KL(pi || ref) (reverse KL)
  per-token loss     l_{i,t} = L_pg(i,t) + beta * k(i,t)
  aggregation        "token":    sum_{i,t} m l / (total completion tokens in the optimizer step)
                     "sequence": sum_i ( sum_t m l / sum_t m ) / (number of completions in the step)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, norm: str = "mean_std", std_eps: float = 1e-4,
                     flat_threshold: float = 1e-6) -> Tuple[torch.Tensor, float]:
    """rewards [N] with the G completions of a prompt CONSECUTIVE (N % G == 0). Returns (advantages [N], fraction of
    groups whose rewards are all (nearly) equal)."""
    if rewards.dim() != 1 or rewards.numel() % group_size != 0:
        raise ValueError(f"rewards {tuple(rewards.shape)} not divisible into groups of {group_size}")
    if norm not in ("mean", "mean_std"):
        raise ValueError("norm must be 'mean' or 'mean_std'")
    r = rewards.detach().to(torch.float32).view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True, unbiased=False)
    flat = std.squeeze(1) < flat_threshold
    adv = r - mean
    if norm == "mean_std":
        adv = adv / (std + std_eps)
    adv = torch.where(flat.unsqueeze(1), torch.zeros_like(adv), adv)
    return adv.reshape(-1), float(flat.float().mean().item())


def group_advantages_sampled_baseline(rewards: torch.Tensor, group_size: int, norm: str = "mean_std", std_eps: float = 1e-4, flat_threshold: float = 1e-6,
                                      greedy_cap: float = 0.0) -> Tuple[torch.Tensor, float]:
    """D-67 (flag `greedy_stats: sampled`): for groups whose LAST member is the greedy completion (`greedy_in_group`). Baseline and std are computed over the group_size - 1 SAMPLED members only; the greedy member
    gets its advantage relative to those statistics, (r_greedy - mean_sampled) / (std_sampled + eps), clipped to +-greedy_cap when greedy_cap > 0. Reason (R8): a looping greedy member (reward ~ -2) inflates the std of the
    whole group 1.3-2.3x and cuts the |advantage| of the sampled members by 25-35 %; with the sampled-only statistics the sampled members keep the advantages they would have without a greedy member.
    A group whose SAMPLED rewards are flat has all advantages 0 (the greedy member too). Returns (advantages [N], fraction of flat groups)."""
    if rewards.dim() != 1 or rewards.numel() % group_size != 0 or group_size < 3:
        raise ValueError(f"rewards {tuple(rewards.shape)} not divisible into groups of {group_size} (>= 3)")
    if norm not in ("mean", "mean_std"):
        raise ValueError("norm must be 'mean' or 'mean_std'")
    r = rewards.detach().to(torch.float32).view(-1, group_size)
    samp = r[:, :-1]
    mean = samp.mean(dim=1, keepdim=True)
    std = samp.std(dim=1, keepdim=True, unbiased=False)
    flat = std.squeeze(1) < flat_threshold
    adv = r - mean
    if norm == "mean_std":
        adv = adv / (std + std_eps)
    if greedy_cap and greedy_cap > 0:
        adv[:, -1] = adv[:, -1].clamp(-float(greedy_cap), float(greedy_cap))
    adv = torch.where(flat.unsqueeze(1), torch.zeros_like(adv), adv)
    return adv.reshape(-1), float(flat.float().mean().item())


def group_advantages_masked(rewards: torch.Tensor, group_size: int, failed: torch.Tensor, mode: str = "zero", norm: str = "mean_std", std_eps: float = 1e-4,
                            flat_threshold: float = 1e-6) -> Tuple[torch.Tensor, float]:
    """D-61 (OFF by default, train_rl `hard_fail_mask`): advantages with the hard-failed members (status != ok: truncated, copy_source, wrong_language ...) taken out of the policy gradient.
      zero     the group baseline and std still include the failed members (so the survivors' advantages are exactly those of `group_advantages`), the failed members get advantage 0.
      exclude  baseline and std are computed over the NON-failed members only (the failures do not move the survivors' advantages either); failed members get advantage 0; a group with < 2
               survivors is flat (all 0).
    NOTE: both remove the NEGATIVE signal against the failure (with the standard rule a -2 sample in a group of ~0.5 rewards has advantage ~ -1.5); `exclude` also removes the positive pressure that
    'succeed rather than fail' puts on the survivors. Returns (advantages [N], fraction of flat groups)."""
    if mode not in ("zero", "exclude"):
        raise ValueError("mode must be 'zero' or 'exclude'")
    if mode == "zero":
        adv, flat = group_advantages(rewards, group_size, norm, std_eps, flat_threshold)
        return torch.where(failed.view(-1), torch.zeros_like(adv), adv), flat
    r = rewards.detach().to(torch.float32).view(-1, group_size)
    ok = (~failed.view(-1, group_size)).to(torch.float32)
    n_ok = ok.sum(dim=1, keepdim=True)
    mean = (r * ok).sum(dim=1, keepdim=True) / n_ok.clamp(min=1.0)
    var = (((r - mean) ** 2) * ok).sum(dim=1, keepdim=True) / n_ok.clamp(min=1.0)
    std = var.sqrt()
    flat = (std.squeeze(1) < flat_threshold) | (n_ok.squeeze(1) < 2)
    adv = (r - mean)
    if norm == "mean_std":
        adv = adv / (std + std_eps)
    adv = adv * ok
    adv = torch.where(flat.unsqueeze(1), torch.zeros_like(adv), adv)
    return adv.reshape(-1), float(flat.float().mean().item())


def scale_last_member(adv: torch.Tensor, group_size: int, scale: float) -> torch.Tensor:
    """D-59: multiply the advantage of the LAST member of every group (the greedy completion of `greedy_in_group`) by `scale`; the group baseline and std (computed over all members)
    are untouched. scale 1 = identity. Reason: a greedy completion of a long document that loops is ONE sequence of up to 7,168 tokens that all push the same way (measured: grad norm
    10-17 with the greedy member, 0.5 without it, shakedown D-59), so it dominates the clipped update."""
    if scale == 1.0:
        return adv
    if adv.dim() != 1 or adv.numel() % group_size != 0:
        raise ValueError(f"advantages {tuple(adv.shape)} not divisible into groups of {group_size}")
    out = adv.clone().view(-1, group_size)
    out[:, -1] = out[:, -1] * float(scale)
    return out.reshape(-1)


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0


KL_LOG_RATIO_MAX = 5.0


def kl_k3_clamped(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """k3 with the log-ratio d = ref_logp - logp clamped at KL_LOG_RATIO_MAX (2026-10-20): exp(d) is unbounded when the policy has made a sampled token far less likely than the
    reference does; two such single-token outliers in phase 2c (steps 607 and 747: KL 0.5 and 1,651, grad norm 220 and 285,381) turned an ordinary step into a clipped-to-noise one and
    tripped the KL guard. Tokens beyond the clamp contribute no KL gradient; everywhere else it equals k3."""
    d = (ref_logp - logp).clamp(max=KL_LOG_RATIO_MAX)
    return torch.exp(d) - d - 1.0


def kl_k2(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    return 0.5 * (ref_logp - logp) ** 2


KL_ESTIMATORS = {"k3": kl_k3, "k3c": kl_k3_clamped, "k2": kl_k2}


def grpo_token_loss(logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor, *, old_logp: Optional[torch.Tensor] = None,
                    ref_logp: Optional[torch.Tensor] = None, kl_coef: float = 0.0, kl_estimator: str = "k3",
                    clip_eps: Optional[float] = None, clip_eps_high: Optional[float] = None,
                    loss_agg: str = "token", normaliser: float = 1.0) -> Dict[str, torch.Tensor]:
    """logp, mask: [B, C]; adv: [B]. Returns a dict with the scalar `loss` (to .backward()) and detached summed
    statistics. `normaliser` = total completion tokens (token agg) or number of completions (sequence agg) of the
    WHOLE optimizer step, so that micro-batch losses simply add up."""
    if logp.shape != mask.shape or adv.shape[0] != logp.shape[0]:
        raise ValueError(f"shape mismatch logp {tuple(logp.shape)} mask {tuple(mask.shape)} adv {tuple(adv.shape)}")
    m = mask.to(logp.dtype)
    a = adv.to(logp.dtype).unsqueeze(1)
    old = logp.detach() if old_logp is None else old_logp
    ratio = torch.exp(logp - old)
    if clip_eps is None:
        pg = -(ratio * a)
        clipped = torch.zeros_like(m)
    else:
        hi = clip_eps if clip_eps_high is None else clip_eps_high
        r_clip = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + hi)
        pg = -torch.minimum(ratio * a, r_clip * a)
        clipped = ((ratio * a > r_clip * a)).to(m.dtype)
    if ref_logp is not None and kl_coef > 0.0:
        kl = KL_ESTIMATORS[kl_estimator](logp, ref_logp)
    else:
        kl = torch.zeros_like(logp)
    per_tok = pg + kl_coef * kl
    if loss_agg == "token":
        loss = (per_tok * m).sum() / normaliser
    elif loss_agg == "sequence":
        loss = ((per_tok * m).sum(1) / m.sum(1).clamp(min=1.0)).sum() / normaliser
    else:
        raise ValueError("loss_agg must be 'token' or 'sequence'")
    return {
        "loss": loss,
        "pg_sum": (pg.detach() * m).sum(), "kl_sum": (kl.detach() * m).sum(), "clip_sum": (clipped * m).sum(),
        "ratio_sum": (ratio.detach() * m).sum(), "n_tokens": m.sum(),
    }
