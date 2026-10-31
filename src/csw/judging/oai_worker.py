"""Polls the OpenAI judge batch every 10 minutes; writes oai/labels.jsonl when it completes. python -m csw.judging.oai_worker (log: oai/worker.log)"""
from csw.paths import DATA_ROOT, subprocess_env
import subprocess, sys, time
from pathlib import Path
HERE = DATA_ROOT; log = (HERE / "oai" / "worker.log").open("a", buffering=1)
while True:
    r = subprocess.run([sys.executable, "-m", "csw.judging.oai_judge", "poll"], cwd=HERE, env=subprocess_env(), capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[0]
    log.write(f"{time.strftime('%H:%M:%S')} {line}\n")
    if line.split(" ")[0] in ("completed", "failed", "expired", "cancelled"): break
    time.sleep(600)
