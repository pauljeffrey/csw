"""Self-training of ONE evaluator: the model labels the ungraded rows, only confident labels are kept (threshold tuned on the dev set), the SAME model continues training on labelled + pseudo-labelled units.
python -m csw.evaluator.selftrain --run runs/v2 --data work/qe_data_v2 --work runs/v2_st [--rounds 2]
Confidence of a unit = agreement of the model with itself over views of the same input: clean input, devices hidden, draft shown (ood), and 4 copies with 10% of the input tokens masked (u = mean over the field's dimensions of the std over views).
Threshold: per field, the largest kept fraction q of the unlabelled units whose dev units (labelled val rows) have mean absolute error <= 0.9 x the field's overall dev error; no such q -> that field gets no pseudo-labels.
A round is accepted only if dev Spearman improves by >= 0.002.  Control: the same model continued on labelled data only for the same number of steps (separates 'more training' from 'pseudo-labels').
Final: one test evaluation of base / control / last accepted round (csw.evaluator.ssl_eval, rule in SSL_PROTOCOL.md: the self-trained model is used only if it beats the base with a bootstrap CI that excludes 0 AND beats the control)."""
from csw.paths import DATA_ROOT, subprocess_env
import argparse, json, random, subprocess, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q

CODE = str(DATA_ROOT)


def unit_views(enc, recs, rnd, k_mask=4, p_mask=0.1):
    """-> items (id lists) and, per unit, the item indices of its views"""
    items, views, meta = [], [], []
    for r in recs:
        for t in ("m", "hm", "ood"):
            main = enc.encode(f"eng_{t}", r, r["eng"], r[t], Q.devices_text(r, t)); v = [len(items)]; items.append(main)
            if t != "m": v.append(len(items)); items.append(enc.encode(f"eng_{t}", r, r["eng"], r[t], ""))
            if t == "ood" and r.get("draft"): v.append(len(items)); items.append(enc.encode(f"eng_{t}", r, r["eng"], r[t], Q.devices_text(r, t), r["draft"]))
            for _ in range(k_mask):
                x = list(main); mid = [i for i in range(1, len(x)) if x[i] not in (enc.sep, enc.cls, enc.pad)]
                for i in rnd.sample(mid, int(p_mask * len(mid))): x[i] = enc.tok.mask_token_id
                v.append(len(items)); items.append(x)
            views.append(v); meta.append((r, t))
    return items, views, meta


def label(model, enc, recs, rnd):
    import numpy as np
    items, views, meta = unit_views(enc, recs, rnd); P = Q.predict(model, enc, items, "cuda", bs=128); out = []
    for v, (r, t) in zip(views, meta):
        dims = Q.FIELD_DIMS[t]; arr = P[v][:, dims]; out.append((r, t, P[v[0]], float(arr.std(0).mean())))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--data", required=True); ap.add_argument("--work", required=True); ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ratio", type=float, default=0.9); ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--lr", type=float, default=1.5e-5); ap.add_argument("--min-gain", type=float, default=0.002)
    ap.add_argument("--max-unl", type=int, default=0); ap.add_argument("--max-train", type=int, default=0); ap.add_argument("--max-test", type=int, default=0, help="smoke-test caps"); a = ap.parse_args()
    import numpy as np, torch
    from transformers import AutoTokenizer
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True); rnd = random.Random(0)
    unl = Q.load(Path(a.data) / "unlabelled.jsonl")[: a.max_unl or None]; va = Q.load(Path(a.data) / "val.jsonl"); cur = Path(a.run); meta = json.load(open(cur / "meta.json")); base_val = prev_val = meta["val_spearman"]; accepted = []
    log = open(work / "selftrain.log", "a")

    def say(m): print(m, flush=True); log.write(m + "\n"); log.flush()
    say(f"base {cur} val spearman {base_val:.4f}")
    for rd in range(1, a.rounds + 1):
        tok = AutoTokenizer.from_pretrained(cur); enc = Q.Enc(tok); model = Q.build_model(json.load(open(cur / "meta.json"))["model"]).cuda(); model.load_state_dict(torch.load(cur / "best.pt", map_location="cuda"))
        U = label(model, enc, unl, rnd); V = label(model, enc, va, rnd); del model; torch.cuda.empty_cache()
        keep = {}; thr = {}
        for t in ("m", "hm", "ood"):
            uu = np.array([u for r, tt, p, u in U if tt == t]); vv = [(u, np.mean([abs(p[i] - r["y"][i]) for i in Q.FIELD_DIMS[t] if r["y"][i] is not None])) for r, tt, p, u in V if tt == t and any(r["y"][i] is not None for i in Q.FIELD_DIMS[t])]
            vu, ve = np.array([x for x, _ in vv]), np.array([e for _, e in vv]); mae_all = ve.mean(); best_q = 0.0
            for q in [x / 10 for x in range(1, 11)]:
                th = np.quantile(uu, q); sel = vu <= th
                if sel.sum() >= 100 and ve[sel].mean() <= a.ratio * mae_all: best_q = q
            thr[t] = (best_q, float(np.quantile(uu, best_q)) if best_q else None, float(mae_all)); say(f"round {rd} field {t}: dev MAE {mae_all:.3f}; kept fraction q*={best_q}")
        rec = {}; n = 0
        for r, t, p, u in U:
            if thr[t][0] and u <= thr[t][1]:
                x = rec.setdefault(r["id"], dict(r, y=[None] * 27, tgts=[])); x["tgts"].append(t); n += 1
                for i in Q.FIELD_DIMS[t]: x["y"][i] = float(round(p[i], 3)) if x["y"][i] is None else round((x["y"][i] + float(p[i])) / 2, 3)
        if n < 1000: say(f"round {rd}: only {n} confident units -> self-training stops (no field passes the dev precision test)"); break
        pf = work / f"pseudo_r{rd}.jsonl"
        with open(pf, "w", encoding="utf-8") as f:
            for x in rec.values(): f.write(json.dumps(x, ensure_ascii=False) + "\n")
        say(f"round {rd}: {n} confident units from {len(rec)} rows -> training")
        out = work / f"round{rd}"
        r_ = subprocess.run([sys.executable, "-u", "-m", "csw.evaluator.train_qe", "--data", a.data, "--out", str(out), "--init", str(cur / "best.pt"), "--extra", str(pf), "--epochs", str(a.epochs), "--lr", str(a.lr), "--seed", str(rd), "--tag", f"round{rd}"] + (["--max-train", str(a.max_train)] if a.max_train else []), cwd=CODE, env=subprocess_env(), stdout=open(work / f"round{rd}.log", "w"), stderr=subprocess.STDOUT)
        if r_.returncode != 0: say(f"round {rd} training FAILED, see {work}/round{rd}.log"); break
        v = json.load(open(out / "meta.json"))["val_spearman"]; say(f"round {rd}: dev Spearman {v:.4f} (previous {prev_val:.4f})")
        if v < prev_val + a.min_gain: say(f"round {rd} rejected (gain < {a.min_gain})"); break
        accepted.append(str(out)); prev_val = v; cur = out
    ctrl = work / "control"
    if not (ctrl / "best.pt").exists():
        subprocess.run([sys.executable, "-u", "-m", "csw.evaluator.train_qe", "--data", a.data, "--out", str(ctrl), "--init", str(Path(a.run) / "best.pt"), "--epochs", str(a.epochs), "--lr", str(a.lr), "--seed", "9", "--tag", "control_labelled_only"] + (["--max-train", str(a.max_train)] if a.max_train else []), cwd=CODE, env=subprocess_env(), stdout=open(work / "control.log", "w"), stderr=subprocess.STDOUT)
    runs = [f"base={a.run}", f"control={ctrl}"] + ([f"selftrained={accepted[-1]}"] if accepted else [])
    cmd = [sys.executable, "-u", "-m", "csw.evaluator.ssl_eval", "--runs", *runs, "--group", "teachers=base", "--group", "random=control", "--data", a.data, "--out", str(work / "test_eval")] + (["--max-test", str(a.max_test)] if a.max_test else [])
    if accepted: cmd += ["--group", "students=selftrained"]
    subprocess.run(cmd, cwd=CODE, env=subprocess_env(), stdout=open(work / "test_eval.log", "w"), stderr=subprocess.STDOUT)
    res = json.load(open(work / "test_eval.json")); adopt = bool(accepted) and res["decision"]["adopt_ssl"]
    final = accepted[-1] if adopt else a.run
    json.dump({"final_run": final, "adopted_selftraining": adopt, "accepted_rounds": accepted, "decision": res["decision"]}, open(work / "final.json", "w"), indent=1)
    say(f"DECISION: self-training adopted = {adopt}; final evaluator = {final}"); say(open(work / "test_eval.txt").read())


if __name__ == "__main__":
    main()
