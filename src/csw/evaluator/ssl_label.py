"""Pseudo-labels from an ensemble of labelled-only teachers, with their disagreement as the uncertainty.
python -m csw.evaluator.ssl_label --teachers runs/t0_s0 runs/t0_s1 runs/t0_s2 --data work/qe_data --out work/qe_ssl
Unit = (row, target field m|hm|ood); input = English source, devices shown, no draft.  mean = ensemble mean over the field's dimensions (None elsewhere); u = mean over those dimensions of the std across teachers.
Also writes calib.json: on the labelled val rows, how well u predicts the ensemble's absolute error (the premise of confidence thresholding)."""
import argparse, json, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--teachers", nargs="+", required=True); ap.add_argument("--data", default="work/qe_data"); ap.add_argument("--out", required=True); a = ap.parse_args()
    import numpy as np, torch
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    unl = Q.load(Path(a.data) / "unlabelled.jsonl"); va = Q.load(Path(a.data) / "val.jsonl"); tok = AutoTokenizer.from_pretrained(a.teachers[0]); enc = Q.Enc(tok)
    mk = lambda recs: [(r, t, enc.encode(f"eng_{t}", r, r["eng"], r[t], Q.devices_text(r, t))) for r in recs for t in ("m", "hm", "ood")]
    U, V = mk(unl), mk(va); preds_u, preds_v = [], []
    for d in a.teachers:
        meta = json.load(open(Path(d) / "meta.json")); model = Q.build_model(meta["model"], meta=meta).cuda(); model.load_state_dict(torch.load(Path(d) / "best.pt", map_location="cuda"))
        preds_u.append(Q.predict(model, enc, [x[2] for x in U], "cuda", bs=128)); preds_v.append(Q.predict(model, enc, [x[2] for x in V], "cuda", bs=128)); del model; torch.cuda.empty_cache(); print("teacher done", d, flush=True)
    PU, PV = np.stack(preds_u), np.stack(preds_v)
    dims = {t: [i for i in Q.FIELD_DIMS[t]] for t in ("m", "hm", "ood")}
    with open(out / "units.jsonl", "w", encoding="utf-8") as f:
        for k, (r, t, _) in enumerate(U):
            mean = [None] * 27
            for i in dims[t]: mean[i] = round(float(PU[:, k, i].mean()), 3)
            f.write(json.dumps({"id": r["id"], "tgt": t, "mean": mean, "u": float(PU[:, k, dims[t]].std(0).mean())}) + "\n")
    cal = {"n_teachers": len(a.teachers), "units": len(U)}
    for t in ("m", "hm", "ood"):
        us, errs, sp1 = [], [], []
        for k, (r, tt, _) in enumerate(V):
            if tt != t: continue
            ys = [(i, r["y"][i]) for i in dims[t] if r["y"][i] is not None]
            if not ys: continue
            us.append(float(PV[:, k, dims[t]].std(0).mean())); errs.append(float(np.mean([abs(PV[:, k, i].mean() - y) for i, y in ys])))
        c = spearmanr(us, errs)[0] if len(us) > 20 else float("nan")
        cal[t] = {"n": len(us), "spearman_u_vs_abs_error": round(float(c), 3), "mean_abs_error": round(float(np.mean(errs)), 3)}
        order = np.argsort(us); h = len(order) // 2
        cal[t]["mae_low_u_half"] = round(float(np.mean(np.array(errs)[order[:h]])), 3); cal[t]["mae_high_u_half"] = round(float(np.mean(np.array(errs)[order[h:]])), 3)
    json.dump(cal, open(out / "calib.json", "w"), indent=1); print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
