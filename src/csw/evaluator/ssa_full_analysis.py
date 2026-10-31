"""Full SSA-MTE test + AfriMTE (eng->X): evaluator vs human z, SSA-COMET, AfriCOMET; regression of each metric onto M1..M6.
python -m csw.evaluator.ssa_full_analysis --segments work/ssa_full.jsonl --preds work/preds_ssa_full.json --comet work/comet_ssa_full.json --out ../../reports/data/evaluator_final_2026-10-30/full"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

EXT = {"SSA-COMET-MTL (ref)": "ssa_mtl_ref", "SSA-COMET-MTL (QE)": "ssa_mtl_qe", "SSA-COMET-QE": "ssa_qe", "AfriCOMET-STL1.1 (ref)": "africomet_stl11_ref", "AfriCOMET-MTL (ref)": "africomet_mtl_ref", "AfriCOMET-QE-STL1.1": "africomet_qe_stl11"}
NAMES = ["M1 faithfulness", "M2 fluency", "M3 orthography", "M4 language identity", "M5 register", "M6 terminology"]


def sp(a, b): return float(spearmanr(a, b)[0])


def dm(x, g):
    x = np.array(x, float); o = x.copy()
    for k in set(g):
        m = np.array([i == k for i in g]); o[m] -= x[m].mean()
    return o


def ci(f, n, B=500):
    r = np.random.default_rng(0); v = [f(r.integers(0, n, n)) for _ in range(B)]; return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser(); [ap.add_argument(f"--{k}", required=True) for k in ("segments", "preds", "comet", "out")]; a = ap.parse_args()
    segs = [json.loads(l) for l in open(a.segments, encoding="utf-8")]; pr = json.load(open(a.preds)); cm = json.load(open(a.comet)); assert pr["ids"] == [s["id"] for s in segs] == cm["ids"]
    keep = np.array([s["human_z"] is not None for s in segs]); segs = [s for s, k in zip(segs, keep) if k]
    lp = [s["lp"] for s in segs]; z = np.array([s["human_z"] for s in segs], float); n = len(segs); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = {k: np.array(v)[keep][:, :6] for k, v in pr["runs"].items()}; Y = {k: np.array(cm[v], float)[keep] for k, v in EXT.items() if cm.get(v)}
    L = [f"# Full SSA-MTE test (16 languages) + AfriMTE eng->X: {n} human-rated segments\n", "Spearman pooled / with per-language-pair means removed (the second removes between-language offsets). 95% bootstrap CI for the pooled value.\n",
         "| scorer | vs human z | CI | within-pair |\n|---|---|---|---|"]
    def row(name, x):
        lo, hi = ci(lambda i: sp(x[i], z[i]), n); L.append(f"| {name} | {sp(x, z):.3f} | [{lo:.3f}, {hi:.3f}] | {sp(dm(x, lp), dm(z, lp)):.3f} |")
    for k, v in P.items(): row(f"our evaluator {k}, mean M1-M6", v.mean(1))
    for k, v in Y.items(): row(k, v)
    best = next((k for k in P if k.endswith("|strict")), next(iter(P))); B = P[best]; Xs = (B - B.mean(0)) / B.std(0)
    L.append(f"\n## Single dimensions of our evaluator ({best}) vs human z and the metrics\n\n| dim | human z | " + " | ".join(Y) + " |\n|---|---|" + "---|" * len(Y))
    for i, nm in enumerate(NAMES): L.append(f"| {nm} | {sp(B[:, i], z):.3f} | " + " | ".join(f"{sp(B[:, i], y):.3f}" for y in Y.values()) + " |")
    L.append(f"\n## Regression of each target on M1..M6 (standardised; cross-validated leaving one language pair out)\n\n| target | R2 in-sample | R2 CV | CV Spearman | mean-M Spearman | NNLS shares % (M1..M6) |\n|---|---|---|---|---|---|")
    pairs = sorted(set(lp)); reg = {}
    for t, y in [("human z", z)] + list(Y.items()):
        ys = (y - y.mean()) / y.std(); A = np.c_[Xs, np.ones(n)]; b = np.linalg.lstsq(A, ys, rcond=None)[0]; r2 = 1 - ((ys - A @ b) ** 2).sum() / (ys ** 2).sum()
        cv = np.zeros(n)
        for g in pairs:
            te = np.array([x == g for x in lp]); cv[te] = A[te] @ np.linalg.lstsq(A[~te], ys[~te], rcond=None)[0]
        cvr = 1 - ((ys - cv) ** 2).sum() / (ys ** 2).sum(); w, _ = nnls(Xs, ys - ys.mean()); sh = 100 * w / max(w.sum(), 1e-9); reg[t] = {"r2": r2, "cv_r2": cvr, "ols_std": b[:6].tolist(), "nnls_share": sh.tolist()}
        L.append(f"| {t} | {r2:.3f} | {cvr:.3f} | {sp(cv, ys):.3f} | {sp(B.mean(1), ys):.3f} | " + ", ".join(f"{s:.0f}" for s in sh) + " |")
    L.append("\n## Per language pair: Spearman with human z\n\n| pair | n | our mean-M | " + " | ".join(Y) + " |\n|---|---|---|" + "---|" * len(Y))
    for g in pairs:
        m = np.array([x == g for x in lp]); L.append(f"| {g} | {m.sum()} | {sp(B[m].mean(1), z[m]):.3f} | " + " | ".join(f"{sp(y[m], z[m]):.3f}" for y in Y.values()) + " |")
    (out / "ssa_full_comparison.md").write_text("\n".join(L)); json.dump(reg, open(out / "regression.json", "w"), indent=1); print("\n".join(L))


if __name__ == "__main__":
    main()
