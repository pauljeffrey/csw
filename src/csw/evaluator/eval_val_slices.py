"""Mean per-dimension Spearman of runs on slices of a val file: records with grader std (old labels) vs strict (new strict labels, never seen by v3).
python -m csw.evaluator.eval_val_slices --runs v3=runs/v3 cft1=runs/cft1 --data cft1"""
import argparse, json, random, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--data", required=True); a = ap.parse_args()
    import numpy as np, torch
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer
    va = Q.load(Path(a.data) / "val.jsonl"); sl = {"old labels (std)": [r for r in va if r.get("grader") != "strict"], "strict labels (held out)": [r for r in va if r.get("grader") == "strict"]}
    tasks = [("eng", "m"), ("eng", "hm"), ("m", "hm"), ("eng", "ood"), ("m", "ood"), ("hm", "ood")]
    for name, d in (x.split("=", 1) for x in a.runs):
        meta = json.load(open(Path(d) / "meta.json")); tok = AutoTokenizer.from_pretrained(d); enc = Q.Enc(tok); enc.grader = bool(meta.get("grader_tag")); model = Q.build_model(meta["model"], meta=meta).cuda(); model.load_state_dict(torch.load(Path(d) / "best.pt", map_location="cuda"))
        for tagname in (["std", "strict"] if enc.grader else ["-"]):
            enc.default_grader = tagname if tagname != "-" else "std"
            for sname, rows in sl.items():
                items, labs = [], []
                for r in rows:
                    for s, t in tasks:
                        if t not in (r.get("tgts") or ["m", "hm", "ood"]) or not r.get(s) or not r.get(t): continue
                        rr = dict(r); rr["grader"] = None if enc.grader else rr.get("grader")
                        ex = Q.make_example(enc, rr, t, random.Random(0), False, task=f"{s}_{t}", use_draft=False); items.append(ex[0]); labs.append(ex[1])
                P = Q.predict(model, enc, items, "cuda", bs=128); sp = []
                for dim in range(27):
                    ys = [l[dim] for l in labs if l[dim] is not None]; ps = [P[k][dim] for k, l in enumerate(labs) if l[dim] is not None]
                    if len(ys) > 50: sp.append(spearmanr(ps, ys)[0])
                print(f"{name:6} tag={tagname:6} slice={sname:26} rows {len(rows):5}  mean per-dimension Spearman {float(np.nanmean(sp)):.4f} over {len(sp)} dimensions", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
