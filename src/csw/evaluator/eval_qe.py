"""Evaluate a trained QE against the two judge LLMs on the held-out test set (rows judged by Flash and, where it can read the language, luna).
python -m csw.evaluator.eval_qe --run runs/qe1 --data work/qe_data --out runs/qe1/eval"""
import argparse, collections, json, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--data", default="work/qe_data"); ap.add_argument("--out", required=True); a = ap.parse_args()
    import numpy as np, torch
    from scipy.stats import spearmanr, pearsonr
    from transformers import AutoTokenizer
    meta = json.load(open(Path(a.run) / "meta.json")); tok = AutoTokenizer.from_pretrained(a.run); enc = Q.Enc(tok)
    model = Q.build_model(meta["model"], meta=meta).cuda(); model.load_state_dict(torch.load(Path(a.run) / "best.pt", map_location="cuda"))
    test = Q.load(Path(a.data) / "test.jsonl"); rnd = None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"meta": meta, "n_test_rows": len(test)}
    def run(with_dev):
        items, idx = [], []
        for k, r in enumerate(test):
            for task, (s, t) in Q.TASKS.items():
                if not r.get(s) or not r.get(t): continue
                dev = Q.devices_text(r, t) if with_dev else ""
                items.append(enc.encode(task, r, r[s], r[t], dev)); idx.append((k, task, t))
        return idx, Q.predict(model, enc, items, "cuda")
    idx, P = run(True); idx0, P0 = run(False)
    def judge_vec(r, j): return r["yj"].get(j)
    def mean_dims(vec, t):
        v = [vec[i] for i in Q.FIELD_DIMS[t] if vec[i] is not None and i not in (24, 25, 26)]
        return sum(v) / len(v) if v else None
    def corr(x, y):
        if len(x) < 8: return None
        return {"n": len(x), "spearman": round(float(spearmanr(x, y)[0]), 3), "pearson": round(float(pearsonr(x, y)[0]), 3), "exact": round(float(np.mean(np.round(x) == np.round(y))), 3), "within1": round(float(np.mean(np.abs(np.round(x) - np.round(y)) <= 1)), 3), "mae": round(float(np.mean(np.abs(np.array(x) - np.array(y)))), 3)}
    tables = {}
    for tag, PP in (("with_devices", P), ("no_devices", P0)):
        by = collections.defaultdict(lambda: collections.defaultdict(lambda: ([], [])))
        for (k, task, t), p in zip(idx, PP):
            r = test[k]; pq = float(np.mean([p[i] for i in Q.FIELD_DIMS[t] if i not in (24, 25, 26)]))
            for j in ("flash", "luna"):
                v = judge_vec(r, j)
                if v is None: continue
                y = mean_dims(v, t)
                if y is None: continue
                by[(task, j)]["overall"][0].append(pq); by[(task, j)]["overall"][1].append(y)
                for i in Q.FIELD_DIMS[t]:
                    if v[i] is not None: by[(task, j)][Q.DIMS[i]][0].append(float(p[i])); by[(task, j)][Q.DIMS[i]][1].append(v[i])
        tables[tag] = {f"{task}|{j}": {d: corr(*xy) for d, xy in dd.items()} for (task, j), dd in by.items()}
    res["qe_vs_judges"] = tables
    # judge vs judge on the rows both graded (the ceiling for agreement with a single judge)
    jj = collections.defaultdict(lambda: ([], []))
    for r in test:
        if "flash" in r["yj"] and "luna" in r["yj"]:
            for t in ("m", "hm", "ood"):
                a_, b_ = mean_dims(r["yj"]["flash"], t), mean_dims(r["yj"]["luna"], t)
                if a_ is not None and b_ is not None: jj[t][0].append(a_); jj[t][1].append(b_)
                for i in Q.FIELD_DIMS[t]:
                    x, y = r["yj"]["flash"][i], r["yj"]["luna"][i]
                    if x is not None and y is not None: jj[Q.DIMS[i]][0].append(x); jj[Q.DIMS[i]][1].append(y)
    res["flash_vs_luna"] = {k: corr(*v) for k, v in jj.items()}
    # QE vs the judges' consensus on rows both graded (compare with flash_vs_luna)
    cons = collections.defaultdict(lambda: ([], []))
    for (k, task, t), p in zip(idx, P):
        r = test[k]
        if "flash" in r["yj"] and "luna" in r["yj"]:
            y = mean_dims(r["yj"][ "flash"], t); z = mean_dims(r["yj"]["luna"], t)
            if y is None or z is None: continue
            q = float(np.mean([p[i] for i in Q.FIELD_DIMS[t] if i not in (24, 25, 26)]))
            cons[task + "|vs_flash"][0].append(q); cons[task + "|vs_flash"][1].append(y); cons[task + "|vs_luna"][0].append(q); cons[task + "|vs_luna"][1].append(z)
    res["qe_vs_each_judge_on_shared_rows"] = {k: corr(*v) for k, v in cons.items()}
    # repaired vs original hm: does the QE (and each judge) prefer the repaired text?
    pr = collections.defaultdict(dict)
    for k, r in enumerate(test):
        if r.get("pair"): pr[r["pair"]["parent"]].setdefault(r["pair"]["role"], k); pr[r["pair"]["parent"]]["c"] = r["pair"]["child"]
    pos = {(k, task): n for n, (k, task, _) in enumerate(idx)}
    win = {"qe": [], "flash": [], "luna": []}
    for par, d in pr.items():
        if "original" in d and "repaired" in d:
            ko, kr = d["original"], d["repaired"]
            q = [(P[pos[(kr, "eng_hm")]][6:15].mean() - P[pos[(ko, "eng_hm")]][6:15].mean())] if (kr, "eng_hm") in pos and (ko, "eng_hm") in pos else []
            if q: win["qe"].append(q[0])
            for j in ("flash", "luna"):
                vo, vr = test[ko]["yj"].get(j), test[kr]["yj"].get(j)
                if vo and vr: win[j].append(mean_dims(vr, "hm") - mean_dims(vo, "hm"))
    res["repaired_minus_original_hm"] = {k: {"n": len(v), "mean_delta": round(float(np.mean(v)), 3) if v else None, "share_repaired_better": round(float(np.mean(np.array(v) > 0)), 3) if v else None} for k, v in win.items()}
    # per-language overall spearman (eng_hm and eng_ood, flash)
    pl = collections.defaultdict(lambda: ([], []))
    for (k, task, t), p in zip(idx, P):
        if task not in ("eng_hm", "eng_ood"): continue
        r = test[k]; v = r["yj"].get("flash")
        if v is None: continue
        y = mean_dims(v, t)
        if y is not None: pl[r["lang"]][0].append(float(np.mean([p[i] for i in Q.FIELD_DIMS[t] if i not in (24, 25, 26)]))); pl[r["lang"]][1].append(y)
    res["per_language_vs_flash_hm_ood"] = {l: corr(*v) for l, v in sorted(pl.items())}
    json.dump(res, open(out / "qe_eval.json", "w"), indent=1)
    # compact markdown
    md = ["# QE vs judge LLMs (test set)\n"]
    def row(name, c): return f"| {name} | {c['n']} | {c['spearman']} | {c['pearson']} | {c['exact']} | {c['within1']} | {c['mae']} |" if c else f"| {name} | - |"
    md.append("| task|judge, overall | n | Spearman | Pearson | exact | within-1 | MAE |\n|---|---|---|---|---|---|---|")
    for key, d in tables["with_devices"].items(): md.append(row(key, d.get("overall")))
    md.append("\n**Judge vs judge (Flash vs luna) on shared rows**\n\n| field | n | Spearman | Pearson | exact | within-1 | MAE |\n|---|---|---|---|---|---|---|")
    for k in ("m", "hm", "ood"): md.append(row(k, res["flash_vs_luna"].get(k)))
    md.append("\n**QE vs each judge on the shared rows**\n\n| task | n | Spearman | Pearson | exact | within-1 | MAE |\n|---|---|---|---|---|---|---|")
    for k, c in res["qe_vs_each_judge_on_shared_rows"].items(): md.append(row(k, c))
    md.append("\n**Repaired minus original hm (H grades)**: " + json.dumps(res["repaired_minus_original_hm"]))
    (out / "qe_eval.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
