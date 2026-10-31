"""What does each group token ([M] [H] [O] [D] [X] [P]) look at?  Internals of the final evaluator, for transparency / alignment work.
python -m csw.evaluator.internals --run runs/ev_final --data work/ev_final --out ../../reports/data/evaluator_internals --n 150 [--device cuda]
Outputs (json + md): (1) attention share per segment by group token (last layer and mean of layers, heads averaged); (2) gradient x embedding saliency per output dimension by segment;
(3) interventions (shuffled source, removed devices, draft on/off, scrambled target) -> mean shift of every output; (4) per-head weight norms / biases, the cosine matrix of the six group-token hidden states, mean scores per dimension.
Segments: task = '<task> | <language> | <pin> | grader' prefix, source, target, devices, draft, group tokens.  No per-dimension tokens exist: each group token feeds ONE head that emits all dimensions of its group."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from csw.evaluator import qe_lib as Q

SEGS = ["task", "source", "target", "devices", "draft", "group_tokens"]


def build(enc, task, rec, s, t, dev, draft):
    head = enc.ids(f"{task} | {Q.lang_name(rec['lang'])} | {rec.get('pin') or '-'}" + (f" | grader: {rec.get('grader') or enc.default_grader}" if enc.grader else ""), 28)
    ids, seg = [enc.cls], [0]
    def add(x, k): ids.extend(x); seg.extend([k] * len(x))
    add(head, 0); add(enc.ids(s, enc.ms), 1); add([enc.sep], 1); add(enc.ids(t, enc.mt), 2); add([enc.sep], 2)
    if dev: add(enc.ids("devices: " + dev, enc.md), 3); add([enc.sep], 3)
    if draft: add(enc.ids("draft: " + draft, enc.ms), 4); add([enc.sep], 4)
    add(enc.groups, 5)
    return ids, seg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=150); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    d = Path(a.run); meta = json.load(open(d / "meta.json")); tok = AutoTokenizer.from_pretrained(d); enc = Q.Enc(tok); enc.grader = bool(meta.get("grader_tag")); enc.default_grader = "strict"
    model = Q.build_model(meta["model"], meta=meta); model.load_state_dict(torch.load(d / "best.pt", map_location="cpu")); model.to(a.device).eval()
    try: model.enc.set_attn_implementation("eager")
    except Exception: model.enc.config._attn_implementation = "eager"
    rnd = random.Random(0); test = [json.loads(l) for l in open(Path(a.data) / "test.jsonl", encoding="utf-8")]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = {"eng_m": "m", "eng_hm": "hm", "m_hm": "hm", "eng_ood": "ood", "m_ood": "ood", "hm_ood": "ood"}
    gi = {g: k for k, (g, _, _) in enumerate(Q.GROUPS)}; res = {"attention": {}, "saliency": {}, "intervention": {}, "scores": {}}
    lay_attn = {}; sal = {}
    pool_by_lang = {}
    for r in test: pool_by_lang.setdefault(r["lang"], []).append(r)
    for task, tgt in tasks.items():
        src = task.split("_")[0]; rows = [r for r in test if r.get(src) and r.get(tgt) and (tgt in (r.get("tgts") or ("m", "hm", "ood")))]; rnd.shuffle(rows); rows = rows[:a.n]
        if not rows: continue
        A_last = np.zeros((6, 6)); A_mean = np.zeros((6, 6)); S = np.zeros((27, 6)); scores = []; base = {}
        def run(items, grad=False):
            ids = torch.tensor([[enc.pad] * (max(len(x) for x, _ in items) - len(x)) + x if False else x + [enc.pad] * (max(len(x) for x, _ in items) - len(x)) for x, _ in items], device=a.device)
            att = (ids != enc.pad).long(); return ids, att
        for r in rows:
            use_draft = tgt == "ood" and bool(r.get("draft"))
            ids, seg = build(enc, task, r, r[src], r[tgt], Q.devices_text(r, tgt), r.get("draft") if use_draft else "")
            x = torch.tensor([ids], device=a.device); att = torch.ones_like(x)
            emb = model.enc.get_input_embeddings()(x).detach().requires_grad_(True)
            o = model.enc(inputs_embeds=emb, attention_mask=att, output_attentions=True); hs = o.last_hidden_state
            pos = [ids.index(g) for g in enc.groups]; outs = torch.cat([h(hs[0, p]) for p, h in zip(pos, model.heads)])
            sc = (outs.detach() * 4 + 1).clamp(1, 5).cpu().numpy(); scores.append(sc)
            sg = torch.tensor(seg, device=a.device)
            for gk in range(6):                                                     # attention: from group token gk (query) to each segment, mean over heads, last layer / mean of layers
                last = o.attentions[-1][0].mean(0)[pos[gk]]; mean = torch.stack([at[0].mean(0)[pos[gk]] for at in o.attentions]).mean(0)
                for s_ in range(6):
                    m = sg == s_
                    if m.any(): A_last[gk, s_] += float(last[m].sum()); A_mean[gk, s_] += float(mean[m].sum())
            for dim in Q.FIELD_DIMS[tgt]:                                          # saliency: |d output / d embedding| x |embedding|, summed over a segment, normalised per example
                g, = torch.autograd.grad(outs[dim], emb, retain_graph=True); sv = (g[0] * emb[0].detach()).abs().sum(-1)
                tot = float(sv.sum()) + 1e-9
                for s_ in range(6):
                    m = sg == s_
                    if m.any(): S[dim, s_] += float(sv[m].sum()) / tot
        n = len(rows); res["attention"][task] = {"last_layer": (A_last / n).round(4).tolist(), "mean_layers": (A_mean / n).round(4).tolist(), "rows": ["[M]", "[H]", "[O]", "[D]", "[X]", "[P]"], "cols": SEGS}
        res["saliency"][task] = {Q.DIMS[dim]: (S[dim] / n).round(4).tolist() for dim in Q.FIELD_DIMS[tgt]}; res["scores"][task] = {Q.DIMS[dim]: float(np.mean([s[dim] for s in scores])) for dim in Q.FIELD_DIMS[tgt]}
        # interventions on the same rows
        def score(rr, s_text=None, t_text=None, dev=None, draft=None):
            ids, _ = build(enc, task, rr, s_text if s_text is not None else rr[src], t_text if t_text is not None else rr[tgt], Q.devices_text(rr, tgt) if dev is None else dev, (rr.get("draft") if draft is None else draft) if tgt == "ood" else "")
            with torch.no_grad(): return Q.predict(model, enc, [ids], a.device)[0]
        iv = {k: [] for k in ("shuffle_source", "no_devices", "draft_on_minus_off", "scramble_target", "drop_half_target")}
        for r in rows[:60]:
            b = score(r, draft=""); oth = rnd.choice(pool_by_lang[r["lang"]])
            iv["shuffle_source"].append(score(r, s_text=oth[src], draft="") - b); iv["no_devices"].append(score(r, dev="", draft="") - b)
            if tgt == "ood" and r.get("draft"): iv["draft_on_minus_off"].append(score(r, draft=r["draft"]) - b)
            w = r[tgt].split(); rnd.shuffle(w); iv["scramble_target"].append(score(r, t_text=" ".join(w), draft="") - b); iv["drop_half_target"].append(score(r, t_text=" ".join(r[tgt].split()[: max(1, len(r[tgt].split()) // 2)]), draft="") - b)
        res["intervention"][task] = {k: {Q.DIMS[dim]: float(np.mean([x[dim] for x in v])) for dim in Q.FIELD_DIMS[tgt]} for k, v in iv.items() if v}
    # parameters: head weights, biases, hidden-state geometry
    W = {}
    for (g, b, e), h in zip(Q.GROUPS, model.heads):
        last = h[-1]; W[g] = {"out_dims": Q.DIMS[b:e], "final_weight_norm": [float(x) for x in last.weight.norm(dim=1)], "final_bias": [float(x) for x in last.bias]}
    res["heads"] = W
    cent = {}
    for task, tgt in tasks.items():
        pass
    H = []
    for r in random.Random(1).sample(test, min(300, len(test))):
        t_ = rnd.choice([t for t, g in tasks.items() if (g in (r.get("tgts") or ("m", "hm", "ood"))) and r.get(t.split("_")[0]) and r.get(g)] or ["eng_m"]); tg = tasks[t_]
        if not (r.get(t_.split("_")[0]) and r.get(tg)): continue
        ids, _ = build(enc, t_, r, r[t_.split("_")[0]], r[tg], Q.devices_text(r, tg), ""); x = torch.tensor([ids], device=a.device)
        with torch.no_grad(): _, hs = model(x, torch.ones_like(x), return_hidden=True)
        H.append(torch.stack([hs[0, ids.index(g)] for g in enc.groups]).cpu().numpy())
    H = np.stack(H); Hc = H.mean(0); Hn = Hc / np.linalg.norm(Hc, axis=1, keepdims=True)
    res["token_geometry"] = {"cosine_of_mean_hidden_states": (Hn @ Hn.T).round(3).tolist(), "hidden_norm": np.linalg.norm(Hc, axis=1).round(2).tolist(), "tokens": ["[M]", "[H]", "[O]", "[D]", "[X]", "[P]"],
                             "mean_pairwise_cosine_across_examples": [float(np.mean([np.dot(H[i, k], H[j, k]) / (np.linalg.norm(H[i, k]) * np.linalg.norm(H[j, k])) for i in range(0, len(H), 7) for j in range(1, len(H), 11)])) for k in range(6)]}
    json.dump(res, open(out / "internals.json", "w"), indent=1)
    L = ["# Evaluator internals\n", "Attention share of each group token (rows) to each input segment (columns), averaged over heads, last layer; rows sum to 1.\n"]
    for task, v in res["attention"].items():
        L.append(f"\n### {task}\n\n| token | " + " | ".join(SEGS) + " |\n|---|" + "---|" * 6)
        for tkn, row in zip(v["rows"], v["last_layer"]): L.append(f"| {tkn} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
    L.append("\n## Saliency share of each output by segment (gradient x embedding, normalised per example)\n")
    for task, v in res["saliency"].items():
        L.append(f"\n### {task}\n\n| output | " + " | ".join(SEGS[:5]) + " |\n|---|---|---|---|---|---|")
        for dim, row in v.items(): L.append(f"| {dim} | " + " | ".join(f"{x:.2f}" for x in row[:5]) + " |")
    L.append("\n## Interventions: mean change of each output (1-5 scale)\n")
    for task, v in res["intervention"].items():
        L.append(f"\n### {task}\n"); 
        for k, dd in v.items(): L.append(f"- {k}: " + ", ".join(f"{dim} {x:+.2f}" for dim, x in dd.items()))
    L.append("\n## Token geometry\n\ncosine of the mean final hidden states (tokens " + ", ".join(res["token_geometry"]["tokens"]) + "):\n")
    for row in res["token_geometry"]["cosine_of_mean_hidden_states"]: L.append("    " + "  ".join(f"{x:+.2f}" for x in row))
    (out / "internals.md").write_text("\n".join(L)); print("\n".join(L)[:3000])


if __name__ == "__main__":
    main()
