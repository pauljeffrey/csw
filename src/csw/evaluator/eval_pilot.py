"""Evaluators vs the four judges on the 299 pilot rows (old Flash, strict Flash, luna, Argon): Spearman of the overall hm / ood score.
python -m csw.evaluator.eval_pilot --runs v3=runs/v3 cft1=runs/cft1 --data qe_data_v3 --labels pilot_labels.json"""
import argparse, json, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--data", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--grader", default="strict"); a = ap.parse_args()
    import numpy as np, torch
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer
    test = {r["id"]: r for r in Q.load(Path(a.data) / "test.jsonl")}; lab = json.load(open(a.labels)); ids = [k for k in lab if k in test]
    md = lambda v, t: (lambda x: sum(x) / len(x) if x else None)([v[i] for i in Q.FIELD_DIMS[t] if v[i] is not None and i not in (24, 25, 26)])
    print(f"{len(ids)} pilot rows;  Spearman of the overall score (hm / ood):"); print(f"{'run':10}" + "".join(f"{j:>22}" for j in ("flash_old", "flash_strict", "luna", "argon")))
    for name, d in (x.split("=", 1) for x in a.runs):
        meta = json.load(open(Path(d) / "meta.json")); tok = AutoTokenizer.from_pretrained(d); enc = Q.Enc(tok); enc.grader = bool(meta.get("grader_tag")); enc.default_grader = a.grader
        model = Q.build_model(meta["model"], meta=meta).cuda(); model.load_state_dict(torch.load(Path(d) / "best.pt", map_location="cuda")); res = {}
        for task, t in (("eng_hm", "hm"), ("eng_ood", "ood")):
            items = [enc.encode(task, test[k], test[k]["eng"], test[k][t], Q.devices_text(test[k], t)) for k in ids]; P = Q.predict(model, enc, items, "cuda", bs=128)
            pred = [float(np.mean([p[i] for i in Q.FIELD_DIMS[t] if i not in (24, 25, 26)])) for p in P]; res[t] = pred
        row = f"{name:10}"
        for j in ("flash_old", "flash_strict", "luna", "argon"):
            s = []
            for t in ("hm", "ood"):
                ok = [(res[t][n], md(lab[k][j], t)) for n, k in enumerate(ids) if md(lab[k][j], t) is not None]; s.append(spearmanr(*zip(*ok))[0])
            row += f"{s[0]:14.3f} / {s[1]:.3f}"
        print(row); del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
