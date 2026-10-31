# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rollout samplers behind one interface (DECISIONS.md D-28): `HFSampler` (the original `model.generate` path) and
`VLLMSampler` (vLLM engine, weights pushed after every optimizer step).

Both return the same `Rollout` (csw.vendor.gemma_mt.rl.sampling) and take the same arguments, so the trainer and the reward do not know which
one produced the ids:

    prompts   -> `Rollout.prompts` repeated `group_size` times, rows grouped (prompt i owns rows i*G .. i*G+G-1)
    completions: token ids exactly as generated, cut AFTER the first stop id (<end_of_turn> or <eos>, included) or all
                 `max_new_tokens` ids when none came (finished = False); nothing is re-tokenised
    post_stop_tokens: tokens after the first stop id that are neither padding nor a stop id (0 by construction for vLLM:
                 the engine stops generating at the stop id; the field is computed anyway)

Log-probs are NOT needed from the sampler: the trainer recomputes log p(sampled ids) with the current policy by teacher forcing
(`losses.completion_logprobs`), `old_logp` is that value detached (ratio == 1, asserted in the loop) and the KL uses the frozen
reference. vLLM log-probs are only returned for the optional consistency check (`check_sampler_logprobs`).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from csw.vendor.gemma_mt.rl.sampling import Rollout, generate_completions
from csw.vendor.gemma_mt.rl.text import cut_at_stop
from csw.vendor.gemma_mt.rl.vllm_engine import RawCompletion, SamplerError, SamplingSpec

SEED_STRIDE = 1024            # the n children of a prompt use seed, seed+1, .. seed+n-1 inside vLLM


def derive_seeds(base_seed: int, key: int, n_prompts: int) -> List[int]:
    """Deterministic per-prompt seeds for one sampling call: a function of (base seed, key = optimizer step or an eval tag, prompt
    index) only. Prompts are SEED_STRIDE apart so the children of two prompts never share a seed."""
    start = (int(base_seed) * 1_000_003 + int(key) * 7_919_137) % (2 ** 31 - 1 - SEED_STRIDE * max(n_prompts, 1))
    return [start + i * SEED_STRIDE for i in range(n_prompts)]


def policy_bf16_weights(policy, dtype=None):
    """(name, tensor) pairs of the policy's parameters in the engine's dtype, tied weights once (lm_head is the embedding)."""
    import torch

    dtype = dtype or torch.bfloat16
    for n, p in policy.named_parameters():
        yield n, p.detach().to(dtype)


class HFSampler:
    """The original sampler: `model.generate` on the training model itself (no copy, always current)."""

    name = "hf"

    def __init__(self, policy, device, autocast_fn):
        self.policy, self.device, self._autocast = policy, device, autocast_fn
        self.sync_seconds = 0.0

    def sync(self, policy, version: int) -> float:
        return 0.0

    def generate(self, prompts, *, group_size, max_new_tokens, temperature, top_p, top_k, stop_ids, pad_id, greedy=False,
                 return_logprobs=False, key: int = 0) -> Rollout:
        return generate_completions(self.policy, prompts, group_size=group_size, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_p=top_p, top_k=top_k, stop_ids=stop_ids, pad_id=pad_id,
                                    device=self.device, greedy=greedy, autocast=self._autocast(), return_logprobs=return_logprobs)

    def close(self) -> None:
        pass


class VLLMSampler:
    """Samples with a vLLM engine whose weights are a bf16 copy of the policy, refreshed by `sync` after every optimizer step."""

    name = "vllm"

    def __init__(self, engine, *, seed: int = 0, sleep_between: bool = False, weight_dtype=None):
        self.engine, self.seed, self.sleep_between = engine, int(seed), bool(sleep_between)
        self.weight_dtype = weight_dtype                    # None = bfloat16 (the engine's dtype)
        self._version: Optional[int] = None
        self.sync_seconds = 0.0
        self.n_syncs = 0
        self.stats: Dict[str, float] = {"tokens": 0.0, "seconds": 0.0, "finish_length": 0.0, "finish_stop": 0.0}

    # ------------------------------------------------------------------------------------------------- weights
    def sync(self, policy, version: int) -> float:
        """Push the policy's weights to the engine unless `version` (the number of optimizer steps taken) is the one already there.
        Returns the seconds spent (0 when nothing was done)."""
        if self._version == version:
            return 0.0
        t0 = time.time()
        info = self.engine.push_weights(policy_bf16_weights(policy, self.weight_dtype))
        self._version = version
        self.n_syncs += 1
        dt = time.time() - t0
        self.sync_seconds = dt
        self.last_sync_info = info
        return dt

    def assume_loaded(self, version: int = 0) -> None:
        """The engine was started from a checkpoint directory and nothing will be pushed (evaluation, not training): declare its weights
        current so that `generate` accepts them. Never used by the trainer, which syncs before every rollout."""
        self._version = int(version)

    def invalidate(self) -> None:
        """The policy changed behind our back (checkpoint loaded): the next generate demands a sync."""
        self._version = None

    # ------------------------------------------------------------------------------------------------- sampling
    def generate(self, prompts: Sequence[Sequence[int]], *, group_size: int, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int, stop_ids: Sequence[int], pad_id: int, greedy: bool = False,
                 return_logprobs: bool = False, key: int = 0) -> Rollout:
        if self._version is None:
            raise SamplerError("the vLLM engine has never been given the policy weights: call sync() before generate()")
        k = 1 if greedy else group_size
        spec = SamplingSpec(n=k, temperature=0.0 if greedy else float(temperature), top_p=float(top_p), top_k=int(top_k),
                            max_tokens=int(max_new_tokens), stop_token_ids=tuple(int(s) for s in stop_ids),
                            seeds=None if greedy else derive_seeds(self.seed, key, len(prompts)), logprobs=bool(return_logprobs))
        t0 = time.time()
        raw = self.engine.generate(prompts, spec)
        dt = time.time() - t0
        if len(raw) != len(prompts):
            raise SamplerError(f"engine returned {len(raw)} prompts' results for {len(prompts)} prompts")
        stops = set(int(s) for s in stop_ids)
        benign = stops | {int(pad_id)}
        comps, fin, lps, post = [], [], [], []
        for p, row in zip(prompts, raw):
            if len(row) != k:
                raise SamplerError(f"engine returned {len(row)} completions for a request of n={k}")
            for c in row:
                ids = list(c.token_ids)
                if not ids:
                    raise SamplerError("engine returned an empty completion")
                if c.finish_reason == "stop" and not any(t in stops for t in ids):
                    # vLLM reports the stop id in `stop_reason` and may leave it out of token_ids: the reward needs it in the ids
                    # (a missing stop id counts as truncation) and the loss needs it as a target
                    if c.stop_token is not None and c.stop_token in stops:
                        ids.append(int(c.stop_token))
                    else:
                        raise SamplerError(f"completion finished with reason 'stop' but contains none of the stop ids {sorted(stops)} "
                                           f"(ends in id {ids[-1]})")
                if c.finish_reason not in ("stop", "length"):
                    raise SamplerError(f"unexpected finish_reason {c.finish_reason!r}")
                if len(ids) > max_new_tokens + 1:
                    raise SamplerError(f"completion has {len(ids)} tokens, more than max_new_tokens {max_new_tokens}")
                kept, finished = cut_at_stop(ids, stop_ids)
                if finished and c.finish_reason == "length" and not (len(ids) == max_new_tokens and ids[-1] in stops):
                    raise SamplerError(f"finish_reason 'length' but the ids contain a stop id before the budget ran out ({len(ids)} ids)")
                # (a stop id that is exactly the last allowed token counts as finished, like HF `generate`; vLLM 0.31 reports it as "stop")
                comps.append(kept)
                fin.append(finished)
                post.append(sum(1 for t in ids[len(kept):] if int(t) not in benign) if finished else 0)
                if c.logprobs is not None:
                    lps.append(list(c.logprobs[: len(kept)]))
        rep_prompts = [list(p) for p in prompts for _ in range(k)]
        self.stats["tokens"] += sum(len(c) for c in comps)
        self.stats["seconds"] += dt
        self.stats["finish_stop"] += sum(fin)
        self.stats["finish_length"] += len(fin) - sum(fin)
        self.last_generate_seconds = dt
        self.last_generate_tokens = sum(len(c) for c in comps)
        return Rollout(rep_prompts, comps, fin, lps if return_logprobs and lps else None, post)

    def sleep(self) -> None:
        """Free the engine's GPU memory until the next sync (its weights are gone, hence the version is forgotten)."""
        if self.sleep_between:
            self.engine.sleep()
            self._version = None

    def close(self) -> None:
        self.engine.close()


class HFBackedEngine:
    """Engine stand-in for CPU tests and `train_rl --smoke-vllm`: a PRIVATE copy of the model, sampled with HF `generate`, that only
    learns the trainer's weights through `push_weights` (by parameter name, like the real engine). It exercises everything around the
    engine - weight versions, seeds, stop/truncation contract, the worker protocol - without vLLM or a GPU."""

    def __init__(self, model_path: str, device: str = "cpu"):
        import torch

        from csw.vendor.gemma_mt.modeling import load_model

        self.model = load_model(model_path, dtype=torch.float32, device=device).eval()
        self.device = device
        self.pushes = 0
        self.generate_calls = 0
        self.closed = False

    def generate(self, prompts, spec: SamplingSpec):
        import torch

        if self.closed:
            raise SamplerError("engine closed")
        self.generate_calls += 1
        if spec.seeds is not None:
            torch.manual_seed(int(spec.seeds[0]))
        greedy = spec.temperature <= 0
        roll = generate_completions(self.model, prompts, group_size=1 if greedy else spec.n, max_new_tokens=spec.max_tokens,
                                    temperature=spec.temperature or 1.0, top_p=spec.top_p, top_k=spec.top_k,
                                    stop_ids=list(spec.stop_token_ids), pad_id=0, device=self.device, greedy=greedy,
                                    return_logprobs=spec.logprobs)
        k = 1 if greedy else spec.n
        rows = []
        for i in range(len(prompts)):
            row = []
            for j in range(k):
                r = i * k + j
                ids = roll.completions[r]
                row.append(RawCompletion(list(ids), "stop" if roll.finished[r] else "length", ids[-1] if roll.finished[r] else None,
                                         roll.sampler_logp[r] if roll.sampler_logp is not None else None))
            rows.append(row)
        return rows

    def push_weights(self, named_tensors: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        import torch

        params = dict(self.model.named_parameters())
        seen = set()
        with torch.no_grad():
            for n, t in named_tensors:
                params[n].copy_(t.to(params[n].dtype))
                seen.add(n)
        missing = sorted(set(params) - seen)
        if missing:
            raise SamplerError(f"weight update left {len(missing)} parameter(s) untouched, e.g. {missing[:3]}")
        self.pushes += 1
        return {"seconds": 0.0}

    def sleep(self) -> None:
        pass

    def wake_up(self) -> None:
        pass

    def info(self) -> Dict[str, Any]:
        return {"engine": "hf_backed", "pushes": self.pushes}

    def close(self) -> None:
        self.closed = True


def build_vllm_sampler(cfg, model_name: str, log_dir: Optional[str] = None) -> VLLMSampler:
    """Engine from the RLConfig fields `vllm_*`. Mode "inprocess": vLLM in the trainer's python; "worker": a child process
    started with `vllm_python` (the interpreter of the environment that has vLLM)."""
    from csw.vendor.gemma_mt.rl.vllm_engine import EngineConfig, VLLMEngine, WorkerEngine
    import os

    ec = EngineConfig(model=model_name, gpu_memory_utilization=cfg.vllm_gpu_memory_utilization, kv_cache_gib=cfg.vllm_kv_cache_gib,
                      max_model_len=cfg.vllm_max_model_len, max_num_seqs=cfg.vllm_max_num_seqs,
                      max_num_batched_tokens=cfg.vllm_max_num_batched_tokens, enforce_eager=cfg.vllm_enforce_eager,
                      enable_sleep_mode=cfg.vllm_sleep, attention_backend=cfg.vllm_attention_backend, seed=cfg.seed,
                      extra=dict(cfg.vllm_extra))
    if cfg.vllm_mode == "inprocess":
        engine = VLLMEngine(ec)
    elif cfg.vllm_mode == "worker":
        engine = WorkerEngine(ec, python=cfg.vllm_python, sync_dir=cfg.vllm_sync_dir,
                              log_path=os.path.join(log_dir, "vllm_worker.log") if log_dir else None,
                              startup_timeout=cfg.vllm_startup_timeout, request_timeout=cfg.vllm_request_timeout)
    else:
        raise ValueError(f"vllm_mode must be 'inprocess' or 'worker', got {cfg.vllm_mode!r}")
    return VLLMSampler(engine, seed=cfg.seed, sleep_between=cfg.vllm_sleep)
