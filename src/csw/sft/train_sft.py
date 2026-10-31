"""SFT of csw on its five tasks (+ eng->m replay).  Single GPU, bf16 autocast over fp32 weights, token-budget batches, loss only on the completion.
python -m csw.sft.train_sft --base work/base --data work/sft --out work/runs/sft1 [--epochs 1] [--lr 5e-5] [--batch-tokens 16384] [--step-tokens 131072]"""
import argparse, json, math, os, random, re, sys, time
from pathlib import Path


def build(tok, path, max_len):
    ids, lab = [], []; dropped = 0
    for l in open(path, encoding="utf-8"):
        e = json.loads(l)
        p = tok(f"<bos><start_of_turn>user\n{e['prompt']}<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)["input_ids"]
        c = tok(e["completion"] + "<end_of_turn>", add_special_tokens=False)["input_ids"]
        if len(p) + len(c) > max_len: dropped += 1; continue
        ids.append(p + c); lab.append([-100] * len(p) + c)
    return ids, lab, dropped


def batches(ids, budget, rnd):
    order = sorted(range(len(ids)), key=lambda i: len(ids[i])); out = []; cur = []; mx = 0
    for i in order:
        mx2 = max(mx, len(ids[i]))
        if cur and mx2 * (len(cur) + 1) > budget: out.append(cur); cur = []; mx2 = len(ids[i])
        cur.append(i); mx = mx2
    if cur: out.append(cur)
    rnd.shuffle(out); return out


def collate(ids, lab, b, pad, dev):
    import torch
    m = max(len(ids[i]) for i in b); x = torch.full((len(b), m), pad); y = torch.full((len(b), m), -100); a = torch.zeros((len(b), m), dtype=torch.long)
    for k, i in enumerate(b):
        n = len(ids[i]); x[k, :n] = torch.tensor(ids[i]); y[k, :n] = torch.tensor(lab[i]); a[k, :n] = 1
    return x.to(dev), y.to(dev), a.to(dev)


def loss_fn(model, x, y, a, chunk=4096):
    import torch
    from torch.utils.checkpoint import checkpoint
    h = model.model(input_ids=x, attention_mask=a).last_hidden_state[:, :-1]; t = y[:, 1:]
    sel = t != -100; h = h[sel]; t = t[sel]; cap = getattr(model.config, "final_logit_softcapping", None)

    def ce(hc, tc):
        lg = model.lm_head(hc).float()
        if cap: lg = torch.tanh(lg / cap) * cap
        return torch.nn.functional.cross_entropy(lg, tc, reduction="sum")
    tot = 0.0
    for s in range(0, h.shape[0], chunk):
        tot = tot + checkpoint(ce, h[s:s + chunk], t[s:s + chunk], use_reentrant=False)
    return tot, int(t.numel())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--batch-tokens", type=int, default=16384); ap.add_argument("--step-tokens", type=int, default=131072)
    ap.add_argument("--max-len", type=int, default=2048); ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--save-every", type=int, default=500); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--patience", type=int, default=0, help="stop when validation loss has not improved for this many evaluations (0 = never)")
    ap.add_argument("--freeze-steps", type=int, default=0, help="up-scaled base: train only the new layers (upscale_info.json) for this many steps, then unfreeze everything"); a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rnd = random.Random(1234)
    tok = AutoTokenizer.from_pretrained(a.base); pad = tok.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev); model.train()
    frozen = False
    if a.freeze_steps > 0:
        new_layers = set(json.load(open(Path(a.base) / "upscale_info.json"))["new_layers"])
        for n, p in model.named_parameters():
            m = re.match(r"model\.layers\.(\d+)\.", n); p.requires_grad = bool(m) and int(m.group(1)) in new_layers
        frozen = True; print(f"frozen phase: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M of {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters trainable (layers {sorted(new_layers)}) for {a.freeze_steps} steps", flush=True)
    tid, tlab, d1 = build(tok, Path(a.data) / "train.jsonl", a.max_len); vid, vlab, d2 = build(tok, Path(a.data) / "val.jsonl", a.max_len)
    print(f"train {len(tid)} (dropped {d1}) val {len(vid)} (dropped {d2}) tokens {sum(map(len, tid))/1e6:.1f}M", flush=True)
    accum = max(1, a.step_tokens // a.batch_tokens)
    ep_batches = []; ne = math.ceil(a.epochs)
    for e in range(ne):
        b = batches(tid, a.batch_tokens, random.Random(1000 + e)); ep_batches += b if e + 1 <= a.epochs else b[: int(len(b) * (a.epochs - e))]
    total_steps = a.max_steps or len(ep_batches) // accum; warm = max(1, int(0.03 * total_steps))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    lr_at = lambda s: a.lr * (s + 1) / warm if s < warm else a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    step = 0; ck = out / "last"
    if (ck / "state.pt").exists():
        st = torch.load(ck / "state.pt", map_location="cpu"); model.load_state_dict(torch.load(ck / "model.pt", map_location="cpu")); opt.load_state_dict(st["opt"]); step = st["step"]; print("resumed at", step, flush=True)

    def evaluate():
        model.eval(); tot = n = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in batches(vid, a.batch_tokens, random.Random(0))[:60]:
                x, y, m = collate(vid, vlab, b, pad, dev); l, c = loss_fn(model, x, y, m); tot += float(l); n += c
        model.train(); return tot / max(1, n)
    best = 9e9; t0 = time.time(); seen = 0; bad = 0
    while step < total_steps:
        if frozen and step >= a.freeze_steps:
            for p in model.parameters(): p.requires_grad = True
            frozen = False; print(f"step {step}: all parameters unfrozen", flush=True)
        grp = ep_batches[step * accum:(step + 1) * accum]
        if not grp: break
        ntok = 0; lsum = 0.0
        for b in grp:
            x, y, m = collate(tid, tlab, b, pad, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l, c = loss_fn(model, x, y, m)
            (l / 40000.0).backward(); lsum += float(l); ntok += c                    # a fixed divisor keeps the sum comparable across steps; rescaled below
        for p in model.parameters():
            if p.grad is not None: p.grad.mul_(40000.0 / max(1, ntok))
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups: g["lr"] = lr_at(step)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1; seen += sum(len(tid[i]) for b in grp for i in b)
        if step % 10 == 0: print(f"step {step}/{total_steps} loss {lsum/max(1,ntok):.4f} gn {float(gn):.2f} lr {lr_at(step):.2e} tok/s {seen/(time.time()-t0):.0f}", flush=True)
        if step % a.eval_every == 0 or step == total_steps:
            vl = evaluate(); print(f"EVAL step {step} val_loss {vl:.4f}", flush=True)
            if vl < best: best = vl; bad = 0; model.save_pretrained(str(out / "best"), safe_serialization=True); tok.save_pretrained(str(out / "best"))
            else: bad += 1
            if a.patience and bad >= a.patience: print(f"early stop at step {step}: no improvement for {bad} evaluations (best {best:.4f})", flush=True); break
        if step % a.save_every == 0:
            ck.mkdir(exist_ok=True); torch.save(model.state_dict(), ck / "model.pt"); torch.save({"opt": opt.state_dict(), "step": step}, ck / "state.pt")
    if not a.no_final: model.save_pretrained(str(out / "final"), safe_serialization=True); tok.save_pretrained(str(out / "final"))
    print("DONE best_val", best, flush=True)


if __name__ == "__main__":
    main()
