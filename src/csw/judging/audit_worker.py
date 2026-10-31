"""Polls the two Gemini audit batches every 12 minutes and writes results when each finishes. python -m csw.judging.audit_worker  (log: exemplars/audit_worker.log)"""
from csw.paths import DATA_ROOT, load_env_file, subprocess_env
import subprocess, sys, time, os
from pathlib import Path
HERE = DATA_ROOT
JOBS = {"flagged": ("--only-flagged", "exemplars/gold_audit_flagged.gemini.jsonl"), "rest": ("--skip-flagged", "exemplars/gold_audit_rest.gemini.jsonl")}
TERMINAL = ("completed", "failed", "expired", "cancelled")
env = dict(os.environ)
load_env_file()
if os.environ.get("OPENROUTER_API_KEY"): env["OPENROUTER_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
log = (HERE / "exemplars" / "audit_worker.log").open("a", buffering=1)
done = set()
while len(done) < len(JOBS):
    for name, (flag, out) in JOBS.items():
        if name in done: continue
        r = subprocess.run([sys.executable, "-m", "csw.judging.repair_gold", "--all", flag, "--out", out, "--batch", "poll"], cwd=HERE, env=subprocess_env(**env), capture_output=True, text=True)
        line = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[0]
        log.write(f"{time.strftime('%H:%M:%S')} {name}: {line}\n")
        if line.split(" ")[0] in TERMINAL: done.add(name)
    if len(done) < len(JOBS): time.sleep(12 * 60)
log.write(f"{time.strftime('%H:%M:%S')} all batches finished\n")
