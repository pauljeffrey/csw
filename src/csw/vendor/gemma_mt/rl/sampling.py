# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sampling completions with HF `generate` and assembling (prompt, completion) tensors for log-prob scoring.

Layout used everywhere in the RL stage (and the one `losses.completion_logprobs` assumes):

    input_ids       [N, P + C]   prompt LEFT-padded to width P, completion RIGHT-padded to width C
    attention_mask  [N, P + C]   0 on both paddings
    completion_mask [N, C]       1 on real completion tokens (the stop id included), 0 on padding
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from csw.vendor.gemma_mt.rl.text import cut_at_stop


@dataclass
class Rollout:
    prompts: List[List[int]]              # one entry per COMPLETION (prompt repeated group_size times)
    completions: List[List[int]]          # ids up to and including the first stop id (or max_new_tokens ids)
    finished: List[bool]
    sampler_logp: Optional[List[List[float]]] = None   # log-prob of each kept token under the sampling distribution
    # per completion: tokens the sampler produced AFTER the first stop id that are neither padding nor another stop id (D-26). HF
    # `generate` pads finished rows with `pad_token_id`, so this is 0 for it; it is non-zero only for a sampler that keeps generating
    # after the end token (eos ignored / min_new_tokens / a custom decoder). The cut below drops those tokens from `completions`.
    post_stop_tokens: List[int] = field(default_factory=list)


def merge_greedy_into_groups(sampled: Rollout, greedy: Rollout, n_sampled: int) -> Rollout:
    """Groups of `n_sampled` temperature-sampled completions per prompt + the prompt's one greedy completion appended as the LAST row of
    each group (group size n_sampled + 1). `sampled` has n_sampled rows per prompt, `greedy` one."""
    n = len(greedy.completions)
    if len(sampled.completions) != n * n_sampled:
        raise ValueError(f"{len(sampled.completions)} sampled rows != {n} prompts x {n_sampled}")

    def weave(a, g, default=None):
        if a is None or g is None:
            return default
        out = []
        for i in range(n):
            out += list(a[i * n_sampled:(i + 1) * n_sampled]) + [g[i]]
        return out

    return Rollout(prompts=weave(sampled.prompts, greedy.prompts), completions=weave(sampled.completions, greedy.completions),
                   finished=weave(sampled.finished, greedy.finished),
                   sampler_logp=weave(sampled.sampler_logp, greedy.sampler_logp),
                   post_stop_tokens=weave(sampled.post_stop_tokens or None, greedy.post_stop_tokens or None, default=[]))


def generate_completions(model: Any, prompts: Sequence[Sequence[int]], *, group_size: int, max_new_tokens: int,
                         temperature: float, top_p: float, top_k: int, stop_ids: Sequence[int], pad_id: int,
                         device, greedy: bool = False, autocast=None, return_logprobs: bool = False) -> Rollout:
    """`group_size` completions per prompt. Rows come out grouped: prompt i owns rows i*G .. i*G+G-1."""
    import torch

    k = 1 if greedy else group_size
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(prompts):                     # LEFT pad: generation appends on the right
        ids[i, width - len(p):] = torch.tensor(list(p), dtype=torch.long)
        mask[i, width - len(p):] = 1
    kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    if greedy:
        # the checkpoint's generation_config carries sampling defaults (top_k=64, top_p=0.95): neutralise them
        kw.update(temperature=None, top_p=None, top_k=None)
    if return_logprobs:
        kw.update(return_dict_in_generate=True, output_scores=True)
    was_training = model.training
    model.eval()
    ctx = autocast if autocast is not None else nullcontext()
    with torch.no_grad(), ctx:
        out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=max_new_tokens,
                             num_return_sequences=k, eos_token_id=list(stop_ids), pad_token_id=pad_id, use_cache=True,
                             **kw)
    if was_training:
        model.train()
    seqs = out.sequences if return_logprobs else out
    trans = None
    if return_logprobs:
        trans = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float().cpu()
    gen = seqs[:, width:].cpu().tolist()
    comps, fin, lps, post = [], [], [], []
    benign = set(int(x) for x in stop_ids) | {int(pad_id)}
    for r, row in enumerate(gen):
        kept, finished = cut_at_stop(row, stop_ids)
        comps.append(kept)
        fin.append(finished)
        post.append(sum(1 for t in row[len(kept):] if int(t) not in benign) if finished else 0)
        if trans is not None:
            lps.append(trans[r, : len(kept)].tolist())
    rep_prompts = [list(p) for p in prompts for _ in range(k)]
    return Rollout(rep_prompts, comps, fin, lps if trans is not None else None, post)


def assemble_batch(prompts: Sequence[Sequence[int]], completions: Sequence[Sequence[int]], pad_id: int, device=None):
    """-> dict(input_ids, attention_mask, completion_mask, prompt_width) as torch tensors."""
    import torch

    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must be parallel lists")
    P = max(len(p) for p in prompts)
    C = max(len(c) for c in completions)
    if C == 0:
        raise ValueError("empty completions")
    n = len(prompts)
    ids = torch.full((n, P + C), pad_id, dtype=torch.long)
    att = torch.zeros((n, P + C), dtype=torch.long)
    cm = torch.zeros((n, C), dtype=torch.long)
    for i, (p, c) in enumerate(zip(prompts, completions)):
        ids[i, P - len(p): P] = torch.tensor(list(p), dtype=torch.long)
        att[i, P - len(p): P] = 1
        if len(c):
            ids[i, P: P + len(c)] = torch.tensor(list(c), dtype=torch.long)
            att[i, P: P + len(c)] = 1
            cm[i, : len(c)] = 1
    out = {"input_ids": ids, "attention_mask": att, "completion_mask": cm, "prompt_width": P}
    if device is not None:
        out = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in out.items()}
    return out
