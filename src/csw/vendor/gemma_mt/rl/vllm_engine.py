# SPDX-License-Identifier: AGPL-3.0-or-later
"""The vLLM side of the rollout sampler (DECISIONS.md D-28): an engine that turns token-id prompts into token-id
completions, and takes the policy's weights after every optimizer step.

Import-light on purpose: the module imports only the standard library at the top level. `vllm` and `torch` are imported
inside `VLLMEngine`, so (a) the CPU test suite can import it, and (b) a worker process started with the python of ANOTHER
virtual environment (the one that has vLLM) can run `python -m csw.vendor.gemma_mt.rl.vllm_engine` with nothing but the package root
on its PYTHONPATH.

Three interchangeable engines (same interface: `generate`, `push_weights`, `close`):

    VLLMEngine     the real thing, in THIS process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`): trainer and engine share one python,
                   one torch and one CUDA context. Weights go GPU -> GPU through `model.load_weights` (no copy to host).
    WorkerEngine   a client for `VLLMEngine` running in a child process with its own interpreter (own venv). Weights travel
                   through ONE shared-memory file on a tmpfs (/dev/shm, mmap'd by both processes, rewritten in place at every sync);
                   requests and answers are small pickles on a unix socket.
    (FakeEngine in the tests)

Failure policy: nothing here falls back to anything. A dead / hung / out-of-memory engine raises `SamplerError`; the trainer
stops. The only way to the HF sampler is the config switch `sampler: hf`.
"""

from __future__ import annotations

import atexit
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class SamplerError(RuntimeError):
    """The vLLM engine failed (start-up, out of memory, died, timed out, produced something that breaks the contract)."""


@dataclass
class EngineConfig:
    model: str                                   # local dir / Hub id with config.json (+ weights; they are overwritten by the first sync)
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.25         # fraction of the TOTAL device memory this engine may use (weights + activations + KV cache);
                                                 # vLLM refuses to start when less than this is FREE, also when kv_cache_gib is set
    kv_cache_gib: Optional[float] = None         # not None: fixed KV-cache size (GiB) instead of "whatever fits in gpu_memory_utilization"
    max_model_len: int = 8192                    # longest prompt + max_new_tokens
    max_num_seqs: int = 256                      # concurrent sequences (the b64 stage samples 32 x 8 = 256)
    max_num_batched_tokens: Optional[int] = None
    enforce_eager: bool = False                  # True: no CUDA graphs / torch.compile (slow, but the quickest to start)
    enable_prefix_caching: bool = True
    enable_sleep_mode: bool = False              # level-2 sleep between rollouts: frees weights + KV cache while the trainer runs
    attention_backend: Optional[str] = None      # None = vLLM's choice (FLASH_ATTN on sm_120, measured); e.g. "TRITON_ATTN", "FLASHINFER"
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)   # passed to vllm.LLM(**extra)


@dataclass
class SamplingSpec:
    n: int = 1                                   # completions per prompt
    temperature: float = 1.0                     # 0 = greedy
    top_p: float = 1.0
    top_k: int = 0                               # 0 = off
    max_tokens: int = 256
    stop_token_ids: Tuple[int, ...] = ()
    seeds: Optional[List[int]] = None            # one seed per PROMPT (the n children use seed, seed+1, ..); None = engine RNG
    logprobs: bool = False                       # return the log-prob of every sampled token (raw logits, i.e. before temperature)


@dataclass
class RawCompletion:
    token_ids: List[int]
    finish_reason: str                           # "stop" | "length" | other vLLM reasons ("abort", "error")
    stop_token: Optional[int] = None             # the stop id that ended it (finish_reason == "stop"), when vLLM reports one
    logprobs: Optional[List[float]] = None


# ---------------------------------------------------------------------------------------------------------- real engine
class VLLMEngine:
    """vLLM `LLM` in this process. Needs `vllm` (and a CUDA GPU)."""

    def __init__(self, cfg: EngineConfig):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"          # engine core in THIS process: callables reach the model unpickled
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        import torch

        n_threads = torch.get_num_threads()
        try:
            from vllm import LLM
        except ImportError as e:  # pragma: no cover - needs the vllm environment
            raise SamplerError(f"vllm is not importable from {sys.executable}: {e}") from e
        self.cfg = cfg
        kw: Dict[str, Any] = dict(model=cfg.model, dtype=cfg.dtype, max_model_len=cfg.max_model_len, max_num_seqs=cfg.max_num_seqs,
                                  enforce_eager=cfg.enforce_eager, enable_prefix_caching=cfg.enable_prefix_caching, seed=cfg.seed,
                                  skip_tokenizer_init=True,          # ids in, ids out: no detokenisation, no re-tokenisation
                                  generation_config="vllm",          # ignore the checkpoint's sampling defaults (top_k 64, top_p 0.95)
                                  enable_sleep_mode=cfg.enable_sleep_mode, disable_log_stats=True)
        kw["gpu_memory_utilization"] = cfg.gpu_memory_utilization   # also checked against the FREE memory when kv_cache_gib is set (vLLM 0.31)
        if cfg.kv_cache_gib is not None:
            kw["kv_cache_memory_bytes"] = int(cfg.kv_cache_gib * 2 ** 30)
        if cfg.max_num_batched_tokens:
            kw["max_num_batched_tokens"] = cfg.max_num_batched_tokens
        if cfg.attention_backend:
            kw["attention_backend"] = cfg.attention_backend
        kw.update(cfg.extra)
        t0 = time.time()
        try:
            self.llm = LLM(**kw)
        except Exception as e:
            raise SamplerError(f"vLLM engine start-up failed: {type(e).__name__}: {e}") from e
        torch.set_num_threads(n_threads)                           # vLLM lowers the torch CPU threads to 1 "for serving"
        self.startup_seconds = time.time() - t0
        self._asleep = False
        self.generated_tokens = 0

    # ------------------------------------------------------------------------------------------------- generation
    def generate(self, prompts: Sequence[Sequence[int]], spec: SamplingSpec) -> List[List[RawCompletion]]:
        from vllm import SamplingParams

        if self._asleep:
            self.wake_up()
        if spec.seeds is not None and len(spec.seeds) != len(prompts):
            raise ValueError("one seed per prompt")
        greedy = spec.temperature <= 0
        params = []
        for i in range(len(prompts)):
            params.append(SamplingParams(
                n=1 if greedy else spec.n, temperature=0.0 if greedy else spec.temperature, top_p=spec.top_p,
                top_k=spec.top_k, max_tokens=spec.max_tokens, stop_token_ids=list(spec.stop_token_ids), detokenize=False,
                ignore_eos=False, skip_special_tokens=False, logprobs=0 if spec.logprobs else None,
                seed=None if spec.seeds is None else int(spec.seeds[i])))
        too_long = [i for i, p in enumerate(prompts) if len(p) + spec.max_tokens > self.cfg.max_model_len]
        if too_long:
            raise SamplerError(f"{len(too_long)} prompt(s) + max_tokens {spec.max_tokens} exceed vllm_max_model_len "
                               f"{self.cfg.max_model_len} (longest prompt {max(len(prompts[i]) for i in too_long)} tokens): raise it")
        try:
            outs = self.llm.generate([{"prompt_token_ids": list(map(int, p))} for p in prompts], params, use_tqdm=False)
        except Exception as e:
            raise SamplerError(f"vLLM generate failed: {type(e).__name__}: {e}") from e
        if len(outs) != len(prompts):
            raise SamplerError(f"vLLM returned {len(outs)} results for {len(prompts)} prompts")
        res: List[List[RawCompletion]] = []
        for o in outs:
            row = []
            for c in o.outputs:
                ids = [int(t) for t in c.token_ids]
                lps = None
                if spec.logprobs:
                    lps = [float(d[t].logprob) for d, t in zip(c.logprobs, ids)]
                sr = c.stop_reason
                row.append(RawCompletion(ids, str(c.finish_reason), int(sr) if isinstance(sr, int) else None, lps))
                self.generated_tokens += len(ids)
            res.append(row)
        return res

    # ------------------------------------------------------------------------------------------------- weights
    def push_weights(self, named_tensors: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        """Copy the given (HF-named) parameters into the engine's model IN PLACE (CUDA graphs stay valid), verify that every
        parameter of the engine's model was written, and drop the prefix cache (cached KV belongs to the old weights)."""
        if self._asleep:
            self.wake_up()
        items = named_tensors

        def _load(model):
            loaded = model.load_weights(iter(items))
            want = {n for n, _ in model.named_parameters()}
            loaded = set(loaded) if loaded is not None else set()
            return sorted(want - loaded)

        t0 = time.time()
        try:
            missing = self.llm.apply_model(_load)[0]
        except Exception as e:
            raise SamplerError(f"vLLM weight update failed: {type(e).__name__}: {e}") from e
        if missing:
            raise SamplerError(f"weight update left {len(missing)} engine parameter(s) untouched, e.g. {missing[:3]}")
        self.llm.reset_prefix_cache(reset_running_requests=True)
        return {"seconds": time.time() - t0}

    def sleep(self) -> None:
        """Level 2: free weights and KV cache (the next `generate` / `push_weights` wakes the engine up; the weights are then
        rewritten by the next sync, which the sampler always does first)."""
        if not self.cfg.enable_sleep_mode:
            raise SamplerError("sleep needs enable_sleep_mode")
        self.llm.sleep(level=2)
        self._asleep = True

    def wake_up(self) -> None:
        self.llm.wake_up()
        self._asleep = False

    def info(self) -> Dict[str, Any]:
        import torch

        free, total = torch.cuda.mem_get_info()
        return {"startup_seconds": self.startup_seconds, "device_free_gib": free / 2 ** 30, "device_total_gib": total / 2 ** 30,
                "torch_allocated_gib": torch.cuda.memory_allocated() / 2 ** 30, "pid": os.getpid()}

    def close(self) -> None:
        llm, self.llm = getattr(self, "llm", None), None
        if llm is None:
            return
        if getattr(self, "_asleep", False):                        # shutting down a sleeping engine aborts inside vLLM's cumem allocator
            try:
                llm.wake_up()
            except Exception:
                pass
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del llm
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------------------- worker process
def default_worker_cmd(python: Optional[str] = None) -> List[str]:
    return [python or sys.executable, "-u", "-m", "csw.vendor.gemma_mt.rl.vllm_engine"]


def _package_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WorkerEngine:
    """Client of a `VLLMEngine` living in a child process (`python -m csw.vendor.gemma_mt.rl.vllm_engine`, any interpreter that has vllm)."""

    def __init__(self, cfg: EngineConfig, *, python: Optional[str] = None, log_path: Optional[str] = None,
                 sync_dir: str = "/dev/shm", startup_timeout: float = 1800.0, request_timeout: float = 3600.0,
                 env: Optional[Dict[str, str]] = None):
        from multiprocessing.connection import Listener

        self.cfg, self.request_timeout = cfg, float(request_timeout)
        self._dir = tempfile.mkdtemp(prefix="gemma_mt_vllm_", dir=sync_dir if os.path.isdir(sync_dir) else None)
        self._weights_path = os.path.join(self._dir, "weights.bin")
        self._layout: Optional[List[Tuple[str, List[int], str, int]]] = None      # (name, shape, dtype, byte offset) of the file
        self._authkey = secrets.token_bytes(16)
        address = os.path.join(self._dir, "sock")
        self._listener = Listener(address, family="AF_UNIX", authkey=self._authkey)
        self.log_path = log_path or os.path.join(self._dir, "worker.log")
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        wenv = dict(os.environ)
        wenv.update(env or {})
        wenv["PYTHONPATH"] = _package_root()                       # nothing of the trainer's environment may shadow the worker's
        wenv["PYTHONDONTWRITEBYTECODE"] = "1"
        wenv["GEMMA_MT_VLLM_AUTHKEY"] = self._authkey.hex()
        self._log = open(self.log_path, "ab")
        cmd = default_worker_cmd(python) + ["--serve", address, "--config-json", json.dumps(asdict(cfg))]
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, env=wenv, cwd=_package_root(),
                                     start_new_session=True)
        atexit.register(self.close)
        self.conn = None
        deadline = time.time() + startup_timeout
        # accept() blocks: do it in a thread so that a worker that dies before connecting is noticed
        import threading

        box: Dict[str, Any] = {}

        def _accept():
            try:
                box["conn"] = self._listener.accept()
            except Exception as e:  # pragma: no cover
                box["err"] = e

        th = threading.Thread(target=_accept, daemon=True)
        th.start()
        while th.is_alive():
            th.join(0.5)
            if self.proc.poll() is not None:
                raise SamplerError(f"vLLM worker exited with code {self.proc.returncode} before connecting\n{self.log_tail()}")
            if time.time() > deadline:
                self.close()
                raise SamplerError(f"vLLM worker did not connect within {startup_timeout:.0f}s\n{self.log_tail()}")
        if "err" in box:
            raise SamplerError(f"vLLM worker handshake failed: {box['err']}")
        self.conn = box["conn"]
        kind, payload = self._recv(deadline - time.time() + 1.0, what="engine start-up")
        if kind != "ready":
            raise SamplerError(f"vLLM worker start-up failed:\n{payload}\n{self.log_tail()}")
        self.worker_info = payload

    # ------------------------------------------------------------------------------------------------- plumbing
    def log_tail(self, n: int = 40) -> str:
        try:
            with open(self.log_path, "rb") as f:
                lines = f.read().decode("utf-8", "replace").splitlines()
            return "--- last lines of " + self.log_path + " ---\n" + "\n".join(lines[-n:])
        except OSError:
            return ""

    def _recv(self, timeout: float, what: str):
        t_end = time.time() + timeout
        while True:
            try:
                if self.conn.poll(1.0):
                    return self.conn.recv()
            except (EOFError, OSError, ConnectionError) as e:
                raise SamplerError(f"vLLM worker connection lost during {what} ({type(e).__name__})\n{self.log_tail()}") from e
            if self.proc.poll() is not None:
                raise SamplerError(f"vLLM worker died (exit code {self.proc.returncode}) during {what}\n{self.log_tail()}")
            if time.time() > t_end:
                raise SamplerError(f"vLLM worker timed out after {timeout:.0f}s during {what}\n{self.log_tail()}")

    def _call(self, op: str, what: str, timeout: Optional[float] = None, **kw):
        if self.conn is None or self.proc.poll() is not None:
            raise SamplerError(f"vLLM worker is not running (exit code {self.proc.poll()})\n{self.log_tail()}")
        try:
            self.conn.send({"op": op, **kw})
        except (OSError, ConnectionError, BrokenPipeError) as e:
            raise SamplerError(f"vLLM worker connection lost while sending {what}: {e}\n{self.log_tail()}") from e
        kind, payload = self._recv(self.request_timeout if timeout is None else timeout, what)
        if kind == "error":
            raise SamplerError(f"vLLM worker failed in {what}:\n{payload}")
        return payload

    # ------------------------------------------------------------------------------------------------- interface
    def generate(self, prompts: Sequence[Sequence[int]], spec: SamplingSpec) -> List[List[RawCompletion]]:
        rows = self._call("generate", "generate", prompts=[list(map(int, p)) for p in prompts], spec=asdict(spec))
        return [[RawCompletion(**c) for c in row] for row in rows]

    def push_weights(self, named_tensors: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        """Write the tensors into the shared-memory file (GPU -> host, no serialisation) and tell the worker to load them."""
        import torch

        t0 = time.time()
        items = [(n, t.detach()) for n, t in named_tensors]
        layout, off = [], 0
        for n, t in items:
            layout.append((n, list(t.shape), str(t.dtype).replace("torch.", ""), off))
            off += (t.numel() * t.element_size() + 63) // 64 * 64
        total = off
        if self._layout is None or [x[:3] for x in self._layout] != [x[:3] for x in layout] or not os.path.exists(self._weights_path):
            with open(self._weights_path, "wb") as f:
                f.truncate(total)
            self._layout = layout
        buf = torch.from_file(self._weights_path, shared=True, size=total, dtype=torch.uint8)
        for (n, t), (_, _, _, o) in zip(items, layout):
            nbytes = t.numel() * t.element_size()
            buf[o: o + nbytes].view(t.dtype).view(t.shape).copy_(t)
        del buf
        t_write = time.time() - t0
        out = dict(self._call("load_shm", "weight update", path=self._weights_path, total=total, layout=layout))
        out["seconds"] = time.time() - t0
        out["write_seconds"] = t_write
        return out

    def sleep(self) -> None:
        self._call("sleep", "sleep")

    def wake_up(self) -> None:
        self._call("wake_up", "wake_up")

    def info(self) -> Dict[str, Any]:
        return self._call("info", "info")

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        try:
            if getattr(self, "conn", None) is not None and proc is not None and proc.poll() is None:
                try:
                    self.conn.send({"op": "close"})
                except Exception:
                    pass
                try:
                    proc.wait(timeout=30)
                except Exception:
                    pass
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except Exception:
                    proc.kill()
            try:
                self._listener.close()
            except Exception:
                pass
            try:
                self._log.close()
            except Exception:
                pass
            import shutil

            shutil.rmtree(getattr(self, "_dir", ""), ignore_errors=True)
            self.conn = None


def serve(address: str, config_json: str) -> int:
    """Worker main loop (runs in the child process)."""
    from multiprocessing.connection import Client

    cfg = EngineConfig(**json.loads(config_json))
    conn = Client(address, family="AF_UNIX", authkey=bytes.fromhex(os.environ["GEMMA_MT_VLLM_AUTHKEY"]))
    try:
        if cfg.extra.get("_engine") == "hf_backed":                # test hook: the worker protocol without vLLM
            from csw.vendor.gemma_mt.rl.vllm_sampler import HFBackedEngine

            engine = HFBackedEngine(cfg.model, device="cpu")
        else:
            engine = VLLMEngine(cfg)
        conn.send(("ready", engine.info()))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
        return 1
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break                                                  # the trainer is gone
        op = msg.get("op")
        if op == "close":
            break
        try:
            if op == "generate":
                spec = SamplingSpec(**{**msg["spec"], "stop_token_ids": tuple(msg["spec"]["stop_token_ids"])})
                out = [[asdict(c) for c in row] for row in engine.generate(msg["prompts"], spec)]
            elif op == "load_shm":
                import torch

                dev = "cpu" if cfg.extra.get("_engine") == "hf_backed" else "cuda"
                buf = torch.from_file(msg["path"], shared=True, size=msg["total"], dtype=torch.uint8)
                layout = msg["layout"]

                def _tensors():                                    # one tensor at a time on the GPU
                    for name, shape, dtype, o in layout:
                        dt = getattr(torch, dtype)
                        n = 1
                        for d in shape:
                            n *= d
                        yield name, buf[o: o + n * torch.empty((), dtype=dt).element_size()].view(dt).view(shape).to(dev)

                out = engine.push_weights(_tensors())
                out["n_tensors"] = len(layout)
                del buf
            elif op == "sleep":
                engine.sleep()
                out = None
            elif op == "wake_up":
                engine.wake_up()
                out = None
            elif op == "info":
                out = engine.info()
            else:
                raise ValueError(f"unknown op {op!r}")
            conn.send(("ok", out))
        except BaseException:
            conn.send(("error", traceback.format_exc()))
    engine.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="vLLM rollout worker (started by WorkerEngine)")
    ap.add_argument("--serve", required=True, help="unix socket address of the trainer")
    ap.add_argument("--config-json", required=True)
    a = ap.parse_args(argv)
    return serve(a.serve, a.config_json)


if __name__ == "__main__":
    sys.exit(main())
