"""Low-end check on held-out natural-bad translation pairs (strict-Flash labels the evaluators never saw): Spearman of the overall M score, and how well the score finds the bad ones (AUC for Flash overall < 3 and < 2.5).
python -m csw.evaluator.eval_lowend --runs v3=runs/v3 cft1=runs/cft1 cft2a=runs/cft2a --data cft2a"""
import argparse, json, random, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


def auc(score, bad):
    import numpy as np
    from scipy.stats import rankdata
    s = np.array(score); b = np.array(bad, bool); r = rankdata(s); n1 = b.sum(); n0 = (~b).sum()
    return float((r[b].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")        # P(score of a bad item < score of a good item) is 1 - this; we report the chance a GOOD item scores higher


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--data", required=True); a = ap.parse_args()
    import numpy as np, torch
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer
    va = [r for r in Q.load(Path(a.data) / "val.jsonl") if r.get("tgts") == ["m"] and r.get("grader") == "strict"]
    y = [np.mean([v for v in r["y"][:6] if v is not None]) for r in va]; print(f"{len(va)} held-out natural-bad / contrast translation pairs; share Flash<3: {100*np.mean([v<3 for v in y]):.0f}%, <2: {100*np.mean([v<2 for v in y]):.0f}%")
    for name, d in (x.split("=", 1) for x in a.runs):
        meta = json.load(open(Path(d) / "meta.json")); tok = AutoTokenizer.from_pretrained(d); enc = Q.Enc(tok); enc.grader = bool(meta.get("grader_tag")); enc.default_grader = "strict"
        model = Q.build_model(meta["model"], meta=meta).cuda(); model.load_state_dict(torch.load(Path(d) / "best.pt", map_location="cuda"))
        items = [enc.encode("eng_m", r, r["eng"], r["m"], "") for r in va]; P = Q.predict(model, enc, items, "cuda", bs=128); p = [float(np.mean(x[:6])) for x in P]
        # good > bad ranking: probability a random good item gets a higher score than a random bad one
        res = []
        for thr in (3.0, 2.5):
            bad = [v < thr for v in y]; res.append(1 - auc(p, bad))
        print(f"{name:7} Spearman(overall M) {spearmanr(p, y)[0]:.3f} | good-above-bad probability: Flash<3 {res[0]:.3f}, Flash<2.5 {res[1]:.3f}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
