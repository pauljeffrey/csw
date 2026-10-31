"""Per-task paired comparison of two QE runs on the held-out test rows (pre-registered for v2 vs v3, 2026-10-29):
  primary   : translation task eng_m, Spearman of the overall M score vs Flash (paired bootstrap over test rows); secondary: vs luna, and each of M1..M6 vs Flash
  no-regression check for hm and ood tasks: the lower end of the 95% interval of (b - a) must stay above -0.02
python -m csw.evaluator.compare_two --a v2=runs/v2 --b v3=runs/v3 --data qe_data_v3 --out runs/v2_vs_v3 [--boot 2000]"""
import argparse, json, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--grader", default="strict", help="grader tag used at inference for runs trained with the tag (std for old-label test sets)"); a = ap.parse_args()
    import numpy as np, torch
    from scipy.stats import rankdata
    from transformers import AutoTokenizer
    test = Q.load(Path(a.data) / "test.jsonl"); R = len(test); tasks = list(Q.TASKS); runs = {a.a.split("=")[0]: a.a.split("=")[1], a.b.split("=")[0]: a.b.split("=")[1]}; names = list(runs)
    def mean_dims(v, t):
        x = [v[i] for i in Q.FIELD_DIMS[t] if v[i] is not None and i not in (24, 25, 26)]
        return sum(x) / len(x) if x else None
    Y = {(t, j): np.full(R, np.nan) for t in tasks for j in ("flash", "luna")}; YD = {(j, i): np.full(R, np.nan) for j in ("flash", "luna") for i in range(6)}
    for k, r in enumerate(test):
        for task, (s, t) in Q.TASKS.items():
            for j in ("flash", "luna"):
                v = r["yj"].get(j)
                if v is not None and r.get(s) and r.get(t):
                    y = mean_dims(v, t)
                    if y is not None: Y[(task, j)][k] = y
        for j in ("flash", "luna"):
            v = r["yj"].get(j)
            if v is not None:
                for i in range(6):
                    if v[i] is not None: YD[(j, i)][k] = v[i]
    P = {n: {} for n in names}; PD = {n: np.full((R, 6), np.nan) for n in names}
    for n, d in runs.items():
        meta = json.load(open(Path(d) / "meta.json")); tok = AutoTokenizer.from_pretrained(d); enc = Q.Enc(tok); enc.grader = bool(meta.get("grader_tag")); enc.default_grader = a.grader; model = Q.build_model(meta["model"], meta=meta).cuda(); model.load_state_dict(torch.load(Path(d) / "best.pt", map_location="cuda"))
        for task, (s, t) in Q.TASKS.items():
            idx = [k for k, r in enumerate(test) if r.get(s) and r.get(t)]
            out = Q.predict(model, enc, [enc.encode(task, test[k], test[k][s], test[k][t], Q.devices_text(test[k], t)) for k in idx], "cuda", bs=128); q = np.full(R, np.nan)
            q[idx] = [float(np.mean([p[i] for i in Q.FIELD_DIMS[t] if i not in (24, 25, 26)])) for p in out]; P[n][task] = q
            if task == "eng_m": PD[n][idx] = out[:, :6]
        del model; torch.cuda.empty_cache()
    def sp(x, y):
        ok = ~np.isnan(x) & ~np.isnan(y); return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]) if ok.sum() > 8 else float("nan")
    def spw(x, y, cnt):
        ok = np.where(~np.isnan(x) & ~np.isnan(y))[0]; ok = np.repeat(ok, cnt[ok]); return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])
    rng = np.random.default_rng(0); cnts = [np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(a.boot)]
    lines = [f"{'task':9}{'judge':7}{names[0]:>8}{names[1]:>8}{'diff':>8}   95% CI of ({names[1]} - {names[0]})"]; res = {}
    def row(label, judge, x0, x1, y):
        d = [spw(x1, y, c) - spw(x0, y, c) for c in cnts]; lo, hi = np.percentile(d, [2.5, 97.5]); a0, a1 = sp(x0, y), sp(x1, y)
        res[f"{label}|{judge}"] = {names[0]: a0, names[1]: a1, "diff": a1 - a0, "ci95": [float(lo), float(hi)]}
        lines.append(f"{label:9}{judge:7}{a0:8.4f}{a1:8.4f}{a1-a0:+8.4f}   [{lo:+.4f}, {hi:+.4f}]")
    for t in ["eng_m", "eng_hm", "m_hm", "eng_ood", "m_ood", "hm_ood"]:
        for j in ("flash", "luna"): row(t, j, P[names[0]][t], P[names[1]][t], Y[(t, j)])
    for i in range(6): row(f"eng_m M{i+1}", "flash", PD[names[0]][:, i], PD[names[1]][:, i], YD[("flash", i)])
    prim = res["eng_m|flash"]; lines.append(""); lines.append(f"PRIMARY (eng_m vs Flash): diff {prim['diff']:+.4f}, CI [{prim['ci95'][0]:+.4f}, {prim['ci95'][1]:+.4f}] -> {'v3 better' if prim['ci95'][0] > 0 else ('v3 worse' if prim['ci95'][1] < 0 else 'no significant difference')}")
    reg = [(k, v["ci95"][0]) for k, v in res.items() if k.split("|")[0] in ("eng_hm", "m_hm", "eng_ood", "m_ood", "hm_ood") and k.endswith("flash")]
    lines.append("NO-REGRESSION on hm/ood vs Flash (lower CI end > -0.02): " + ", ".join(f"{k.split('|')[0]} {'ok' if lo > -0.02 else 'FAIL'} ({lo:+.3f})" for k, lo in reg))
    Path(a.out + ".txt").write_text("\n".join(lines)); json.dump(res, open(a.out + ".json", "w"), indent=1); print("\n".join(lines))


if __name__ == "__main__":
    main()
