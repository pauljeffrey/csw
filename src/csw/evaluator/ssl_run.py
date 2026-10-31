"""Unattended orchestrator for the semi-supervised QE study (protocol: qe/SSL_PROTOCOL.md).  Resumable: a stage whose DONE marker exists is skipped.
python -m csw.evaluator.ssl_run [--budget-gb 20] [--epochs 3]"""
from csw.paths import DATA_ROOT, subprocess_env
import argparse, glob, json, os, re, subprocess, sys, time
from pathlib import Path

W = Path(os.environ.get("CSW_WORK", "work")); CODE = str(DATA_ROOT); PY = os.environ.get("CSW_PYTHON", sys.executable); QD = W / "qe_data"; RUNS = W / "qe_runs"; SSL = W / "qe_ssl"; LOGS = W / "qe_logs"


def note(msg):
    line = f"[qe-ssl {time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True)
    with open(W / "overnight_notes.log", "a") as f: f.write(line + "\n")
    logs = sorted(glob.glob(str(W / "queue_logs" / "[0-9]*.log")), key=os.path.getmtime)
    if logs:
        with open(logs[-1], "a") as f: f.write(line + "\n")


def free_gb():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split(",")
    return (float(o[1]) - float(o[0])) / 1024


class Sched:
    def __init__(self, budget, per_job): self.maxpar = max(1, int(budget // per_job)); self.per_job = per_job; self.run = {}

    def submit(self, name, args, out):
        marker = Path(out) / "DONE"
        if marker.exists(): return
        while len(self.run) >= self.maxpar or free_gb() < self.per_job + 1: self.reap(); time.sleep(20)
        Path(out).mkdir(parents=True, exist_ok=True); log = open(LOGS / f"{name}.log", "w")
        p = subprocess.Popen([PY, "-u", "-m"] + args, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, env=subprocess_env(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")); self.run[name] = (p, marker); note(f"start {name} (parallel {len(self.run)}/{self.maxpar})")

    def reap(self):
        for n, (p, m) in list(self.run.items()):
            if p.poll() is not None:
                note(f"{n} finished rc={p.returncode}" + ("" if p.returncode == 0 else "  <-- FAILED, see qe_logs/" + n + ".log"))
                if p.returncode == 0: m.write_text("ok")
                del self.run[n]

    def wait(self):
        while self.run: self.reap(); time.sleep(20)


def run_group(sch, jobs):
    for attempt in range(5):
        for name, args, out in jobs: sch.submit(name, args, out)
        sch.wait()
        if all((Path(o) / "DONE").exists() for _, _, o in jobs): return
        note(f"some jobs failed (GPU memory is shared with the SFT queue); retrying in 2 min (attempt {attempt + 2}/5)"); time.sleep(120)
    note("JOBS FAILED REPEATEDLY: see qe_logs"); sys.exit(1)


def val_of(d): return json.load(open(Path(d) / "meta.json"))["val_spearman"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget-gb", type=float, default=20); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--per-job-gb", type=float, default=0); a = ap.parse_args()
    for d in (RUNS, SSL, LOGS): d.mkdir(parents=True, exist_ok=True)
    tq = lambda name, extra=(), seed=0, tag="": ["csw.evaluator.train_qe", "--data", str(QD), "--out", str(RUNS / name), "--seed", str(seed), "--epochs", str(a.epochs), "--tag", tag or name, *extra]
    per_job = a.per_job_gb
    if not (RUNS / "smoke" / "DONE").exists():
        note("smoke test (400 labelled rows, 1 epoch) to check the code path and measure GPU memory per run")
        s = Sched(100, 1); s.submit("smoke", tq("smoke", ["--max-train", "400", "--eval-every", "20"], tag="smoke")[0:0] + ["csw.evaluator.train_qe", "--data", str(QD), "--out", str(RUNS / "smoke"), "--max-train", "400", "--epochs", "1", "--eval-every", "20"], RUNS / "smoke"); s.wait()
        if not (RUNS / "smoke" / "DONE").exists(): note("SMOKE TEST FAILED: see qe_logs/smoke.log"); sys.exit(1)
    if not per_job:
        g = [float(x) for x in re.findall(r"gpu (\d+\.\d+)GB", open(LOGS / "smoke.log").read())]; per_job = (max(g) if g else 12) + 2; note(f"measured peak {max(g) if g else '?'} GB per encoder run -> {per_job:.1f} GB budgeted each")
    sch = Sched(a.budget_gb, per_job); note(f"scheduler: budget {a.budget_gb} GB, {sch.maxpar} parallel run(s)")
    teachers = [f"t0_s{s}" for s in range(3)]
    run_group(sch, [(t, tq(t, seed=int(t[-1])), RUNS / t) for t in teachers])
    if not (SSL / "DONE").exists():
        note("pseudo-labelling the unlabelled pool with the teacher ensemble")
        r = subprocess.run([PY, "-u", "-m", "csw.evaluator.ssl_label", "--teachers", *[str(RUNS / t) for t in teachers], "--data", str(QD), "--out", str(SSL)], cwd=CODE, env=subprocess_env(), stdout=open(LOGS / "label.log", "w"), stderr=subprocess.STDOUT)
        if r.returncode != 0: note("LABELLING FAILED: see qe_logs/label.log"); sys.exit(1)
        (SSL / "DONE").write_text("ok")
    note("uncertainty check on val: " + json.dumps({k: v for k, v in json.load(open(SSL / "calib.json")).items() if k in ("m", "hm", "ood")}))
    qs = (0.25, 0.5, 0.75, 1.0)
    for q in qs:
        f = SSL / f"pseudo_q{q}.jsonl"
        if not f.exists(): subprocess.run([PY, "-m", "csw.evaluator.ssl_select", "--units", str(SSL / "units.jsonl"), "--data", str(QD), "--q", str(q), "--mode", "conf", "--out", str(f)], cwd=CODE, env=subprocess_env(), check=True, stdout=open(LOGS / f"select_q{q}.log", "w"))
    run_group(sch, [(f"ssl_q{q}_s0", tq(f"ssl_q{q}_s0", ["--extra", str(SSL / f"pseudo_q{q}.jsonl")], seed=0), RUNS / f"ssl_q{q}_s0") for q in qs])
    vals = {q: val_of(RUNS / f"ssl_q{q}_s0") for q in qs if (RUNS / f"ssl_q{q}_s0" / "DONE").exists()}; qstar = max(vals, key=vals.get)
    note(f"threshold selection on val: {json.dumps(vals)} -> q* = {qstar}; teachers val {[round(val_of(RUNS / t), 4) for t in teachers]}")
    rf = SSL / f"pseudo_random_q{qstar}.jsonl"
    if not rf.exists(): subprocess.run([PY, "-m", "csw.evaluator.ssl_select", "--units", str(SSL / "units.jsonl"), "--data", str(QD), "--q", str(qstar), "--mode", "random", "--out", str(rf)], cwd=CODE, env=subprocess_env(), check=True, stdout=open(LOGS / "select_random.log", "w"))
    run_group(sch, [(f"rand_q{qstar}_s0", tq(f"rand_q{qstar}_s0", ["--extra", str(rf)], seed=0), RUNS / f"rand_q{qstar}_s0")] + [(f"ssl_q{qstar}_s{s}", tq(f"ssl_q{qstar}_s{s}", ["--extra", str(SSL / f"pseudo_q{qstar}.jsonl")], seed=s), RUNS / f"ssl_q{qstar}_s{s}") for s in (1, 2)])
    students = [f"ssl_q{qstar}_s{s}" for s in range(3)]; rand = [f"rand_q{qstar}_s0"]
    allruns = teachers + [f"ssl_q{q}_s0" for q in qs if q != qstar] + students + rand
    allruns = [r for r in allruns if (RUNS / r / "DONE").exists()]
    note("single-use test evaluation of all runs")
    cmd = [PY, "-u", "-m", "csw.evaluator.ssl_eval", "--runs", *[f"{r}={RUNS / r}" for r in allruns], "--group", "teachers=" + ",".join(teachers), "--group", "students=" + ",".join(s for s in students if s in allruns), "--group", "random=" + ",".join(rand), "--data", str(QD), "--out", str(SSL / "test_eval")]
    r = subprocess.run(cmd, cwd=CODE, env=subprocess_env(), stdout=open(LOGS / "test_eval.log", "w"), stderr=subprocess.STDOUT)
    if r.returncode != 0: note("TEST EVAL FAILED: see qe_logs/test_eval.log"); sys.exit(1)
    res = json.load(open(SSL / "test_eval.json")); adopt = res["decision"]["adopt_ssl"]
    pool = [s for s in students if s in allruns] if adopt else teachers; final = max(pool, key=lambda n: val_of(RUNS / n))
    json.dump({"final_run": str(RUNS / final), "adopted_ssl": adopt, "decision": res["decision"], "chosen_by": "best validation Spearman within the adopted family", "q_star": qstar}, open(RUNS / "final.json", "w"), indent=1)
    note(f"DECISION: adopt SSL = {adopt}; final evaluator = {final}.  Details: qe_ssl/test_eval.txt")
    note(open(SSL / "test_eval.txt").read())


if __name__ == "__main__":
    main()
