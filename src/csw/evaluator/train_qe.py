"""python -m csw.evaluator.train_qe --data work/qe_data --out runs/qe/t0_s0 --seed 0 [--model jhu-clsp/mmBERT-base] [--epochs 3] [--bs 32] [--lr 4e-5]
                       [--extra pseudo.jsonl --extra-weight 0.5]      semi-supervised student: labelled examples + pseudo-labelled units (records carry "tgts": the target fields that were selected)
Validation (the dev set that selects checkpoints and the SSL threshold) = mean Spearman over the dimensions, devices shown, no draft."""
import argparse, json, math, random, sys, time
from collections import defaultdict
from pathlib import Path
from csw.evaluator import qe_lib as Q


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0]) if len(a) > 5 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="work/qe_data"); ap.add_argument("--out", required=True); ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=4e-5); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--extra", default=""); ap.add_argument("--extra-weight", type=float, default=0.5); ap.add_argument("--tag", default=""); ap.add_argument("--grader-tag", action="store_true", help="add the grader tag (std / strict) to every input"); ap.add_argument("--dim-tokens", action="store_true", help="one special token + one scalar head per output dimension instead of six group tokens"); ap.add_argument("--patience", type=int, default=0, help="stop after this many evaluations without a new best (0 = run the whole schedule)"); ap.add_argument("--human-dev", default="", help="human-rated dev rows (csw.evaluator.build_human); their Spearman(mean M1-M6, human z) is added to the selection score"); ap.add_argument("--init", default="", help="start from this state dict (self-training rounds continue the same model)"); a = ap.parse_args()
    import numpy as np, torch
    from transformers import AutoTokenizer
    dev = "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rnd = random.Random(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model); gids = Q.add_group_tokens(tok, Q.DIM_GROUPS if a.dim_tokens else None); ARCH = {"arch": "dim" if a.dim_tokens else "group", "vocab": len(tok), "group_ids": gids}; enc = Q.Enc(tok); enc.grader = a.grader_tag
    tr = Q.load(Path(a.data) / "train.jsonl"); va = Q.load(Path(a.data) / "val.jsonl")
    if a.max_train: tr = random.Random(123).sample(tr, min(a.max_train, len(tr)))      # random subset (the file is ordered by judging batch)
    ex_rec = Q.load(a.extra) if a.extra else []
    pool = defaultdict(list)
    for r in tr:
        if not r.get("tgts"): pool[r["lang"]].append(r)            # translation-only (replay) records cannot serve as another row's source for hm / ood negatives
    fpool = {}
    for l_, rs in pool.items():
        ws = [w for r in rnd.sample(rs, min(300, len(rs))) for w in r["m"].split()]; fpool[l_] = [w for w in ws if len(w) > 2][:3000]
    model = Q.build_model(a.model, vocab=len(tok), gids=gids).to(dev)
    if a.init: model.load_state_dict(torch.load(a.init, map_location=dev)); print('initialised from', a.init, flush=True)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr}, {"params": (model.heads if hasattr(model, "heads") else model.head).parameters(), "lr": a.head_lr}], weight_decay=0.01, betas=(0.9, 0.98))
    val_items = [Q.make_example(enc, r, t, rnd, False, task=f"{s}_{t}", use_draft=False) for r in va for s, t in [("eng", "m"), ("eng", "hm"), ("m", "hm"), ("eng", "ood"), ("m", "ood"), ("hm", "ood")] if t in (r.get("tgts") or ("m", "hm", "ood")) and r.get(s) and r.get(t)]
    units = [(r, t, 1.0, True) for r in tr for t in (r.get("tgts") or ("m", "hm", "ood"))] ; units_x = [(r, t, a.extra_weight, False) for r in ex_rec for t in r["tgts"]]
    n_ex = len(units) + len(units_x); steps_total = a.epochs * math.ceil(n_ex / a.bs); warm = int(0.05 * steps_total); step = 0; best = -1; t0 = time.time(); bad = 0; stop = False
    print(f"labelled units {len(units)} pseudo units {len(units_x)} steps {steps_total} seed {a.seed}", flush=True)
    lr0 = [g["lr"] for g in opt.param_groups]

    hdev = [json.loads(l) for l in open(a.human_dev, encoding="utf-8")] if a.human_dev else []
    hitems = [Q.make_example(enc, {**r, "h": None}, "m", rnd, False, task="eng_m")[0] for r in hdev]

    def evaluate():
        P = Q.predict(model, enc, [x[0] for x in val_items], dev); sp = []; mae = []
        if hdev: Ph = Q.predict(model, enc, hitems, dev); hs = spearman(Ph[:, :6].mean(1), [r["hz"] for r in hdev]); print(f"   human-dev spearman {hs:.4f}", flush=True)
        else: hs = 0.0
        for dim in range(27):
            ys = [l[1][dim] for l in val_items if l[1][dim] is not None]; ps = [P[k][dim] for k, l in enumerate(val_items) if l[1][dim] is not None]
            if len(ys) > 50: sp.append(spearman(ps, ys)); mae.append(float(np.mean(np.abs(np.array(ps) - np.array(ys)))))
        model.train(); return float(np.nanmean(sp)) + hs, float(np.mean(mae))
    model.train()
    for ep in range(a.epochs):
        if stop: break
        exs = units + units_x; rnd.shuffle(exs); chunks = [exs[i:i + a.bs * 50] for i in range(0, len(exs), a.bs * 50)]
        for ch in chunks:
            if stop: break
            ch = [(*Q.make_example(enc, r, t, rnd, True, pool=pool if lab else None, fpool=fpool if lab else None), w) for r, t, w, lab in ch]; ch.sort(key=lambda e: len(e[0])); bats = [ch[i:i + a.bs] for i in range(0, len(ch), a.bs)]; rnd.shuffle(bats)
            for b in bats:
                if stop: break
                ids, att, y, w, hy, hw = Q.collate(b, enc.pad, dev, human=True)
                fac = (step + 1) / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps_total - warm)))
                for g, l0 in zip(opt.param_groups, lr0): g["lr"] = l0 * max(0.02, fac)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(ids, att)
                pf = p.float(); loss = ((torch.nn.functional.huber_loss(pf, y, delta=0.25, reduction="none") * w).sum() + (torch.nn.functional.huber_loss(pf[:, :6].mean(1), hy, delta=0.25, reduction="none") * hw).sum()) / (w.sum() + hw.sum()).clamp(min=1)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 50 == 0: print(f"ep {ep} step {step}/{steps_total} loss {float(loss):.4f} {step*a.bs/(time.time()-t0):.0f} ex/s gpu {torch.cuda.max_memory_reserved()/1e9:.1f}GB", flush=True)
                if step % a.eval_every == 0 or step == steps_total:
                    s, m = evaluate(); print(f"EVAL step {step} mean val spearman {s:.4f} mae {m:.4f}", flush=True)
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": max(best, s), "bad": bad}, out / "last.pt")      # resumable checkpoint (model + optimizer + step)
                    bad = 0 if s > best + 1e-4 else bad + 1
                    if a.patience and bad >= a.patience: print(f"EARLY STOP: no improvement for {bad} evaluations", flush=True); stop = True
                    if s > best: best = s; torch.save(model.state_dict(), out / "best.pt"); tok.save_pretrained(str(out)); json.dump({"model": a.model, "step": step, "val_spearman": s, "val_mae": m, "seed": a.seed, "extra": a.extra, "extra_weight": a.extra_weight, "tag": a.tag, "grader_tag": a.grader_tag, **ARCH}, open(out / "meta.json", "w"))
    s, m = evaluate(); print(f"FINAL val spearman {s:.4f} mae {m:.4f} best {best:.4f}", flush=True)
    if s > best: torch.save(model.state_dict(), out / "best.pt"); tok.save_pretrained(str(out)); json.dump({"model": a.model, "step": step, "val_spearman": s, "val_mae": m, "seed": a.seed, "extra": a.extra, "extra_weight": a.extra_weight, "tag": a.tag, "grader_tag": a.grader_tag, **ARCH}, open(out / "meta.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
