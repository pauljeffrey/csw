"""Single-use test evaluation of QE runs against the two judge LLMs, with paired bootstrap over test rows, and the pre-registered SSL decision (qe/SSL_PROTOCOL.md).
python -m csw.evaluator.ssl_eval --runs t0_s0=runs/t0_s0 ... --group teachers=t0_s0,t0_s1,t0_s2 --group students=... --group random=... --out qe_ssl/test_eval"""
import argparse, collections, json, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--group", action="append", default=[]); ap.add_argument("--data", default="work/qe_data")
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--max-test", type=int, default=0); a = ap.parse_args()
    import numpy as np, torch
    from scipy.stats import rankdata
    from transformers import AutoTokenizer
    test = Q.load(Path(a.data) / "test.jsonl")[: a.max_test or None]; R = len(test); tasks = list(Q.TASKS)
    runs = dict(x.split("=", 1) for x in a.runs); groups = {g.split("=")[0]: g.split("=")[1].split(",") for g in a.group}

    def mean_dims(vec, t):
        v = [vec[i] for i in Q.FIELD_DIMS[t] if vec[i] is not None and i not in (24, 25, 26)]
        return sum(v) / len(v) if v else None
    Y = {(t, j): np.full(R, np.nan) for t in tasks for j in ("flash", "luna")}
    for k, r in enumerate(test):
        for task, (s, t) in Q.TASKS.items():
            for j in ("flash", "luna"):
                v = r["yj"].get(j)
                if v is not None and r.get(s) and r.get(t):
                    y = mean_dims(v, t)
                    if y is not None: Y[(task, j)][k] = y
    PRED, META, VARIANT = {}, {}, {}
    for name, d in runs.items():
        meta = json.load(open(Path(d) / "meta.json")); META[name] = meta; tok = AutoTokenizer.from_pretrained(d); enc = Q.Enc(tok)
        model = Q.build_model(meta["model"], meta=meta).cuda(); model.load_state_dict(torch.load(Path(d) / "best.pt", map_location="cuda")); PRED[name] = {}; VARIANT[name] = {}
        for task, (s, t) in Q.TASKS.items():
            idx = [k for k, r in enumerate(test) if r.get(s) and r.get(t)]
            for tag, dev, dr in (("main", True, False), ("nodev", False, False), ("draft", True, True)):
                if tag == "draft" and t != "ood": continue
                items = [enc.encode(task, test[k], test[k][s], test[k][t], Q.devices_text(test[k], t) if dev else "", test[k].get("draft", "") if dr else "") for k in idx]
                P = Q.predict(model, enc, items, "cuda", bs=128); q = np.full(R, np.nan)
                q[idx] = [float(np.mean([p[i] for i in Q.FIELD_DIMS[t] if i not in (24, 25, 26)])) for p in P]
                (PRED[name] if tag == "main" else VARIANT[name]).__setitem__(task if tag == "main" else (task, tag), q)
        del model; torch.cuda.empty_cache(); print("predicted", name, flush=True)

    def fast_sp(x, y):
        return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]) if len(x) > 8 else float("nan")

    def primary(name, j, counts=None, pred=None):
        sps = []
        for task in tasks:
            p = (pred or PRED[name])[task]; y = Y[(task, j)]; ok = np.where(~np.isnan(p) & ~np.isnan(y))[0]
            if counts is not None: ok = np.repeat(ok, counts[ok])
            sps.append(fast_sp(p[ok], y[ok]))
        return float(np.nanmean(sps)), sps
    res = {"n_test_rows": R, "runs": {}, "groups": {}, "pairs": {}}
    for name in runs:
        pf, spf = primary(name, "flash"); pl, spl = primary(name, "luna")
        mae = float(np.nanmean([np.nanmean(np.abs(PRED[name][t] - Y[(t, "flash")])) for t in tasks]))
        dv = {f"{t}|{tag}": primary(name, "flash", pred={**PRED[name], t: VARIANT[name][(t, tag)]})[0] for t in ("eng_ood", "m_ood", "hm_ood") for tag in ("draft", "nodev")}
        dv.update({f"{t}|nodev": primary(name, "flash", pred={**PRED[name], t: VARIANT[name][(t, "nodev")]})[0] for t in tasks if (t, "nodev") in VARIANT[name]})
        res["runs"][name] = {"val_spearman": META[name]["val_spearman"], "val_mae": META[name].get("val_mae"), "test_primary_flash": pf, "test_secondary_luna": pl, "per_task_flash": dict(zip(tasks, spf)), "per_task_luna": dict(zip(tasks, spl)), "mae_vs_flash": mae, "primary_with_variant_on_one_task": dv}
    rng = np.random.default_rng(0); boots = {n: [] for n in runs}
    for _ in range(a.boot):
        counts = np.bincount(rng.integers(0, R, R), minlength=R)
        for n in runs: boots[n].append(primary(n, "flash", counts)[0])
    B = {n: np.array(v) for n, v in boots.items()}
    for g, names in groups.items():
        res["groups"][g] = {"runs": names, "val_mean": float(np.mean([META[n]["val_spearman"] for n in names])), "test_primary_mean": float(np.mean([res["runs"][n]["test_primary_flash"] for n in names])), "test_luna_mean": float(np.mean([res["runs"][n]["test_secondary_luna"] for n in names]))}
    def diff(g1, g2):
        d = np.mean([B[n] for n in groups[g1]], 0) - np.mean([B[n] for n in groups[g2]], 0)
        return {"mean_diff": float(res["groups"][g1]["test_primary_mean"] - res["groups"][g2]["test_primary_mean"]), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    for g1 in groups:
        for g2 in groups:
            if g1 < g2 or g1 == g2: continue
            res["pairs"][f"{g1}-{g2}"] = diff(g1, g2)
    dec = {"adopt_ssl": False, "reason": "students or teachers missing"}
    if "students" in groups and "teachers" in groups:
        d = diff("students", "teachers"); c1 = bool(d["ci95"][0] > 0)
        c2 = bool(res["groups"]["students"]["test_primary_mean"] > res["groups"]["random"]["test_primary_mean"]) if "random" in groups else True
        c3 = bool((res["groups"]["students"]["val_mean"] - res["groups"]["teachers"]["val_mean"]) > 0 and d["mean_diff"] > 0)
        dec = {"adopt_ssl": bool(c1 and c2 and c3), "students_beat_teachers_ci_excludes_0": c1, "students_beat_control": c2, "val_and_test_both_improve": c3, "diff_students_minus_teachers": d}
    res["decision"] = dec; json.dump(res, open(a.out + ".json", "w"), indent=1)
    lines = [f"{'run':26}{'val sp':>8}{'test(flash)':>12}{'test(luna)':>11}{'MAE':>7}"] + [f"{n:26}{r['val_spearman']:8.4f}{r['test_primary_flash']:12.4f}{r['test_secondary_luna']:11.4f}{r['mae_vs_flash']:7.3f}" for n, r in res["runs"].items()]
    lines += [f"group {g}: val {v['val_mean']:.4f} test {v['test_primary_mean']:.4f} luna {v['test_luna_mean']:.4f}" for g, v in res["groups"].items()] + [f"pair {k}: {json.dumps(v)}" for k, v in res["pairs"].items()] + ["DECISION: " + json.dumps(dec)]
    Path(a.out + ".txt").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
