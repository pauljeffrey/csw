"""Client of qe_worker: starts the worker in the COMET venv once, scores batches synchronously."""
from csw.paths import DATA_ROOT, subprocess_env
import json, subprocess, os, sys, threading


class QEClient:
    def __init__(self, python=None, cwd=None):
        # python: interpreter of the COMET environment (CSW_COMET_PYTHON); cwd: working directory of the worker (default: data root)
        python = python or os.environ.get("CSW_COMET_PYTHON", sys.executable); cwd = cwd or str(DATA_ROOT); self.log = os.environ.get("CSW_QE_LOG", "qe_worker.log")
        env = subprocess_env(TOKENIZERS_PARALLELISM="false"); env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        self.p = subprocess.Popen([python, "-m", "csw.rl.qe_worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(self.log, "a"), text=True, cwd=cwd, env=env, bufsize=1); self.n = 0
        while True:                                                                         # the worker prints non-JSON progress lines from libraries; wait for the ready line
            l = self.p.stdout.readline()
            if not l: raise RuntimeError(f"QE worker died while loading (see {self.log})")
            if l.startswith("{") and '"ready"' in l: break

    def __call__(self, pairs):
        self.n += 1; self.p.stdin.write(json.dumps({"id": self.n, "pairs": pairs}, ensure_ascii=False) + "\n"); self.p.stdin.flush()
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError("QE worker died")
            if l.startswith("{") and '"scores"' in l: return json.loads(l)["scores"]
