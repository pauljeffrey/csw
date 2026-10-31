"""Our evaluator vs human ratings, SSA-COMET and AfriCOMET on the 300 SSA-MTE / AfriMTE segments (en -> amh yor kin hau swh ibo som xho), plus a regression of each external metric on our six translation dimensions M1..M6.
python -m csw.evaluator.ssa_analysis --segments work/ssa300.jsonl --preds work/preds_ssa300_a.json work/preds_ssa300_b.json --out ../../reports/data/evaluator_final_2026-10-30"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr


def demean(x, groups):
    x = np.array(x, float); out = x.copy()
    for g in set(groups):
        m = np.array([gg == g for gg in groups]); out[m] -= x[m].mean()
    return out


def sp(a, b): return float(spearmanr(a, b)[0])


def boot_ci(f, n, B=1000, seed=0):
    rng = np.random.default_rng(seed); v = [f(rng.integers(0, n, n)) for _ in range(B)]; return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--segments", required=True); ap.add_argument("--preds", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    segs = [json.loads(l) for l in open(a.segments, encoding="utf-8")]; ids = [s["id"] for s in segs]; lp = [s["lp"] for s in segs]; n = len(segs); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    preds = {}
    for f in a.preds:
        d = json.load(open(f)); assert d["ids"] == ids
        for k, v in d["runs"].items(): preds[k] = np.array(v)[:, :6]
    ext = {"human z": "human_z", "SSA-COMET-MTL (ref)": "ssa_mtl_ref", "SSA-COMET-MTL-QE": "ssa_mtl_qe", "SSA-COMET-QE": "ssa_qe", "AfriCOMET-STL1.1 (ref)": "africomet_stl11_ref", "AfriCOMET-MTL (ref)": "africomet_mtl_ref"}
    Y = {k: np.array([s[v] for s in segs], float) for k, v in ext.items()}; L = []
    L.append(f"# Final evaluator on {n} human-rated MT segments (SSA-MTE x4 and AfriMTE x4 language pairs)\n")
    L.append("## 1. Spearman of the evaluator's average of M1..M6 with human ratings and with the other metrics (pooled / per-pair means removed)\n")
    L.append("| scorer | vs human z | vs SSA-COMET-MTL ref | vs SSA-COMET-MTL-QE | vs AfriCOMET-STL1.1 ref | vs AfriCOMET-MTL ref |\n|---|---|---|---|---|---|")
    def row(name, x):
        cells = [f"{sp(x, Y[k]):.3f} / {sp(demean(x, lp), demean(Y[k], lp)):.3f}" for k in ("human z", "SSA-COMET-MTL (ref)", "SSA-COMET-MTL-QE", "AfriCOMET-STL1.1 (ref)", "AfriCOMET-MTL (ref)")]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    for k, P in preds.items(): row(f"evaluator {k} (mean M1-M6)", P.mean(1))
    for k in ("SSA-COMET-MTL (ref)", "SSA-COMET-MTL-QE", "AfriCOMET-STL1.1 (ref)", "AfriCOMET-MTL (ref)"):
        L.append(f"| {k} | {sp(Y[k], Y['human z']):.3f} / {sp(demean(Y[k], lp), demean(Y['human z'], lp)):.3f} | - | - | - | - |")
    best = "cft2b|strict" if "cft2b|strict" in preds else list(preds)[-1]; P = preds[best]; Xs = (P - P.mean(0)) / P.std(0)
    L.append(f"\n## 2. Each of M1..M6 on its own against human z and the metrics ({best})\n\n| dimension | human z | SSA-COMET ref | AfriCOMET-STL1.1 ref |\n|---|---|---|---|")
    names = ["M1 faithfulness", "M2 fluency/grammar", "M3 orthography", "M4 language identity", "M5 register", "M6 terminology/names"]
    for i, nm in enumerate(names): L.append(f"| {nm} | {sp(P[:, i], Y['human z']):.3f} | {sp(P[:, i], Y['SSA-COMET-MTL (ref)']):.3f} | {sp(P[:, i], Y['AfriCOMET-STL1.1 (ref)']):.3f} |")
    L.append(f"\n## 3. Regression of each target on M1..M6 ({best}); standardised predictors; 8-fold cross-validation grouped by language pair\n")
    L.append("| target | in-sample R2 | cross-validated R2 | CV Spearman | simple-average Spearman | " + " | ".join(f"w{i + 1}%" for i in range(6)) + " |\n|---|---|---|---|---|" + "---|" * 6)
    reg = {}
    for tname in ("human z", "SSA-COMET-MTL (ref)", "AfriCOMET-STL1.1 (ref)", "AfriCOMET-MTL (ref)"):
        y = Y[tname]; ys = (y - y.mean()) / y.std(); A = np.c_[Xs, np.ones(n)]
        beta, *_ = np.linalg.lstsq(A, ys, rcond=None); r2 = 1 - ((ys - A @ beta) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
        w, _ = nnls(Xs, ys - ys.mean()); share = 100 * w / max(w.sum(), 1e-9)
        pairs = sorted(set(lp)); cv = np.zeros(n)
        for g in pairs:
            te = np.array([x == g for x in lp]); tr = ~te; b, *_ = np.linalg.lstsq(A[tr], ys[tr], rcond=None); cv[te] = A[te] @ b
        cvr2 = 1 - ((ys - cv) ** 2).sum() / ((ys - ys.mean()) ** 2).sum(); reg[tname] = {"r2": float(r2), "cv_r2": float(cvr2), "beta_std": beta[:6].tolist(), "nnls_share_pct": share.tolist()}
        L.append(f"| {tname} | {r2:.3f} | {cvr2:.3f} | {sp(cv, ys):.3f} | {sp(P.mean(1), ys):.3f} | " + " | ".join(f"{s:.0f}" for s in share) + " |")
    L.append("\nOLS standardised coefficients (can be negative; M dimensions are correlated with each other):\n")
    for tname, r in reg.items(): L.append(f"- {tname}: " + ", ".join(f"{nm.split()[0]} {b:+.2f}" for nm, b in zip(names, r["beta_std"])))
    L.append("\n## 4. Per language pair: Spearman of evaluator mean M1-M6 with human z\n")
    for g in sorted(set(lp)):
        m = np.array([x == g for x in lp]); L.append(f"- {g} (n={m.sum()}): evaluator {sp(P[m].mean(1), Y['human z'][m]):.3f}; SSA-COMET ref {sp(Y['SSA-COMET-MTL (ref)'][m], Y['human z'][m]):.3f}; AfriCOMET-STL1.1 {sp(Y['AfriCOMET-STL1.1 (ref)'][m], Y['human z'][m]):.3f}")
    (out / "ssa_comet_comparison.md").write_text("\n".join(L)); json.dump(reg, open(out / "regression.json", "w"), indent=1); print("\n".join(L))


if __name__ == "__main__":
    main()
