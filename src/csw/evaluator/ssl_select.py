"""Select pseudo-labelled units: the q lowest-uncertainty fraction per field ("conf") or the same number chosen at random ("random" control).
python -m csw.evaluator.ssl_select --units work/qe_ssl/units.jsonl --data work/qe_data --q 0.5 --mode conf --out work/qe_ssl/pseudo_q0.5.jsonl"""
import argparse, collections, json, math, random, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--units", required=True); ap.add_argument("--data", default="work/qe_data"); ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--mode", choices=("conf", "random"), default="conf"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); a = ap.parse_args()
    unl = {r["id"]: r for r in Q.load(Path(a.data) / "unlabelled.jsonl")}; by = collections.defaultdict(list)
    for l in open(a.units, encoding="utf-8"):
        u = json.loads(l); by[u["tgt"]].append(u)
    rnd = random.Random(a.seed); rec = {}; n = collections.Counter()
    for t, us in by.items():
        k = math.ceil(a.q * len(us)); us = sorted(us, key=lambda x: x["u"])[:k] if a.mode == "conf" else rnd.sample(us, k)
        for u in us:
            r = rec.setdefault(u["id"], dict(unl[u["id"]], y=[None] * 27, tgts=[]))
            r["tgts"].append(t); n[t] += 1
            for i, v in enumerate(u["mean"]):
                if v is not None: r["y"][i] = v if r["y"][i] is None else round((r["y"][i] + v) / 2, 3)      # X1 is shared by hm and ood
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rec.values(): f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(a.mode, "q", a.q, "units", dict(n), "rows", len(rec))


if __name__ == "__main__":
    main()
