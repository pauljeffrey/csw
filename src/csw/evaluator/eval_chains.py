"""Do the evaluators order held-out corruption chains correctly?  original > mild > moderate > severe, per kind (translation, hm, ood).  Flash's own ordering of the same chains is shown as the reference.
python -m csw.evaluator.eval_chains --runs v3=runs/v3 cft2a=runs/cft2a --chains cft2b/heldout_chains.jsonl"""
import argparse, collections, json, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


def kind(r): return "translation" if r.get("tgts") == ["m"] else ("hm" if r.get("tgts") == ["hm"] else "ood")


def order_acc(chains, key):
    ok = tot = 0
    for c in chains.values():
        ks = sorted(c)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = c[ks[i]].get(key), c[ks[j]].get(key)
                if a is None or b is None: continue
                tot += 1; ok += a > b
    return ok / max(1, tot), tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--chains", required=True); a = ap.parse_args()
    import numpy as np, torch
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in open(a.chains, encoding="utf-8")]; tgt = {"translation": ("eng_m", "m", None), "hm": ("eng_hm", "hm", "hm"), "ood": ("eng_ood", "ood", "ood")}
    flash = {}
    chains = {k: collections.defaultdict(dict) for k in tgt}
    for r in rows:
        k = kind(r); lv = r.get("chain_level", 0) or 0; t = tgt[k][1]; dims = Q.FIELD_DIMS[t]
        fv = [r["y"][i] for i in dims if r["y"][i] is not None and i not in (24, 25, 26)]
        chains[k][r.get("chain_base") or r["id"]][lv] = {"flash": sum(fv) / len(fv) if fv else None, "_r": r}
    print("held-out chains:", {k: len(v) for k, v in chains.items()})
    for k in tgt: print(f"  {k:12} Flash ordering accuracy {100*order_acc(chains[k], 'flash')[0]:.0f}% of {order_acc(chains[k], 'flash')[1]} level pairs")
    for name, d in (x.split("=", 1) for x in a.runs):
        meta = json.load(open(Path(d) / "meta.json")); tok = AutoTokenizer.from_pretrained(d); enc = Q.Enc(tok); enc.grader = bool(meta.get("grader_tag")); enc.default_grader = "strict"
        model = Q.build_model(meta["model"], meta=meta).cuda(); model.load_state_dict(torch.load(Path(d) / "best.pt", map_location="cuda")); out = []
        for k, (task, t, _) in tgt.items():
            items, refs = [], []
            for cb, c in chains[k].items():
                for lv, e in c.items():
                    r = e["_r"]; items.append(enc.encode(task, r, r["eng"], r[t], Q.devices_text(r, t) if t != "m" else "")); refs.append(e)
            P = Q.predict(model, enc, items, "cuda", bs=128); dims = Q.FIELD_DIMS[t]
            for e, p in zip(refs, P): e[name] = float(np.mean([p[i] for i in dims if i not in (24, 25, 26)]))
            acc, n = order_acc(chains[k], name); out.append(f"{k} {100*acc:.0f}%")
        print(f"{name:7} ordering accuracy: " + ", ".join(out), flush=True); del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
