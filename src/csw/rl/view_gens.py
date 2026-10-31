"""Readable view of the generations logged during RL.  python -m csw.rl.view_gens OUT_DIR [--follow] [--monitor] [--task eng_hm] [--n 6]"""
import argparse, json, time, textwrap
from pathlib import Path


def show(r):
    pens = " ".join(f"{k}={v}" for k, v in (r.get("pens") or {}).items()) or "-"
    print(f"\n=== step {r['step']} | {r.get('kind', 'monitor-greedy')} | {r['task']} {r['lang']}{'+' + r['switch'] if r.get('switch') else ''} | reward {r['reward']} | evaluator {r.get('evaluator')} | pens: {pens}{' | FORMAT-FAIL' if r.get('gate') else ''}")
    print("SOURCE:", textwrap.shorten(r["source"], 400)); print("OUTPUT:", r["completion"][:900])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--follow", action="store_true"); ap.add_argument("--monitor", action="store_true"); ap.add_argument("--task", default=""); ap.add_argument("--n", type=int, default=6); a = ap.parse_args()
    f = Path(a.out) / ("monitor_gens.jsonl" if a.monitor else "gens_log.jsonl"); pos = 0
    rows = [json.loads(l) for l in open(f, encoding="utf-8")] if f.exists() else []; rows = [r for r in rows if not a.task or r["task"] == a.task]
    for r in rows[-a.n:]: show(r)
    pos = f.stat().st_size if f.exists() else 0
    while a.follow:
        time.sleep(3)
        if f.exists() and f.stat().st_size > pos:
            with open(f, encoding="utf-8") as fh:
                fh.seek(pos)
                for l in fh:
                    r = json.loads(l)
                    if not a.task or r["task"] == a.task: show(r)
                pos = fh.tell()


if __name__ == "__main__":
    main()
