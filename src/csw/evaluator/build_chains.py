"""Corruption chains for the evaluator's low end.  A chain = original + up to three severity levels of ONE typical failure; the corrupted items are graded by Flash (labels are measured, never assumed).
  translation pairs : 13 operators (omission, phrase skipping, name dropping, repetition loops, missing diacritics, word order, number / negation changes, wrong-language and English injection, truncation, addition)
  hm / ood fields   : the repo's operators (foreign sentences, third-language injection, over / under mixing, literal or scrambled idioms, flat text, diacritics, truncation) + loops
python -m csw.evaluator.build_chains --out work/chains --tr-chains 3750 --hm-chains 750 --ood-chains 750"""
from csw.paths import DATA_ROOT
import argparse, collections, copy, json, random, re, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import corrupt_ladder as CL
from csw.judging.corrupt_ladder import OPS, sub, words, join


def p_drop_phrases(t, lv, rnd):
    w = words(t)
    if len(w) < 8: return None
    for _ in range((1, 2, 4)[lv - 1]):
        n = rnd.randint(2, 4)
        if len(w) <= n + 3: break
        i = rnd.randrange(0, len(w) - n); del w[i:i + n]
    return " ".join(w)


def p_drop_names(t, lv, rnd):
    w = words(t); idx = [i for i, x in enumerate(w) if i > 0 and x[:1].isupper()] + [i for i, x in enumerate(w) if any(c.isdigit() for c in x)]
    idx = sorted(set(idx))
    if not idx: return None
    k = (1, max(1, len(idx) // 2), len(idx))[lv - 1]; drop = set(rnd.sample(idx, min(k, len(idx))))
    return " ".join(x for i, x in enumerate(w) if i not in drop)


def p_loop(t, lv, rnd):
    w = words(t)
    if len(w) < 8: return None
    keep = max(3, int(len(w) * (0.6, 0.4, 0.25)[lv - 1])); phrase = w[max(0, keep - rnd.randint(3, 5)):keep]
    return " ".join(w[:keep] + phrase * (3, 6, 12)[lv - 1])


for name, dim, field, fn in (("M1_phrase_drop", "M1", "m_text", p_drop_phrases), ("M6_name_drop", "M6", "m_text", p_drop_names), ("M2_loop", "M2", "m_text", p_loop),
                             ("H_loop", "H2", "hm", p_loop), ("O_loop", "O3", "ood", p_loop)):
    OPS[name] = (dim, field, sub(field, lambda r, lv, rnd, c, fn=fn, field=field: fn(r[field], lv, rnd)))
M_OPS = [k for k, v in OPS.items() if v[1] in ("m_text", "eng") and k.startswith("M")]
HM_OPS = [k for k, v in OPS.items() if v[1] == "hm"]; OOD_OPS = [k for k, v in OPS.items() if v[1] == "ood"]


def chains_for(row, ops, rnd, ctx, want_levels=2):
    for op in rnd.sample(ops, len(ops)):
        ch = CL.make_chain(row, op, rnd, ctx)
        if len(ch) >= want_levels: return op, ch
    return None, []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tr-chains", type=int, default=3750); ap.add_argument("--hm-chains", type=int, default=750); ap.add_argument("--ood-chains", type=int, default=750)
    ap.add_argument("--tr-pool", default="work/tr_base_pool.jsonl"); ap.add_argument("--unl", default="work/qe_data/unlabelled.jsonl"); ap.add_argument("--used", default="work/new_rows_ids.json"); ap.add_argument("--rows", default="work/post_rows"); ap.add_argument("--heldout", type=float, default=0.12); ap.add_argument("--seed", type=int, default=9); a = ap.parse_args()
    rnd = random.Random(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); st = collections.Counter()
    # ---- translation chains
    pool = [json.loads(l) for l in open(a.tr_pool, encoding="utf-8")]; rnd.shuffle(pool); per = collections.Counter(); take = []
    cap = max(1, a.tr_chains // 34 + 1)
    for r in pool:
        if per[r["lang"]] < cap: per[r["lang"]] += 1; take.append(r)
    take = take[: a.tr_chains]
    rows_tr = [{"id": f"trc{i}", "lang": r["lang"], "eng": r["eng"], "m_text": r["m"], "hm": r["m"], "ood": r["m"], "scenario": {"switch": "English"}, "origin": r["origin"]} for i, r in enumerate(take)]
    ctx = {"pool_by_lang": collections.defaultdict(list), "lexicon": None, "foreign_other": {}}
    for r in rows_tr: ctx["pool_by_lang"][r["lang"]].append(r)
    tr_items = []
    for r in rows_tr:
        op, ch = chains_for(r, M_OPS, rnd, ctx)
        if not ch: st["tr_no_chain"] += 1; continue
        held = rnd.random() < a.heldout; st["tr_chains"] += 1
        tr_items.append({"uid": r["id"] + "~orig", "base": r["id"], "op": op, "level": 0, "lang": r["lang"], "eng": r["eng"], "m": r["m_text"], "heldout": held, "origin": "chain"})
        for c in ch: tr_items.append({"uid": f"{r['id']}~{op}~L{c['level']}", "base": r["id"], "op": op, "level": c["level"], "lang": r["lang"], "eng": c["row"]["eng"], "m": c["row"]["m_text"], "heldout": held, "origin": "chain"})
    with open(out / "tr_items.jsonl", "w", encoding="utf-8") as f:
        for x in tr_items: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    # ---- hm / ood chains from fresh rows
    used = set(json.load(open(a.used))); cand = [json.loads(l) for l in open(a.unl, encoding="utf-8")]; cand = [r["id"] for r in cand if r["id"] not in used]; rnd.shuffle(cand)
    need = set(cand[: (a.hm_chains + a.ood_chains) * 2]); raw = {}
    for f in sorted(Path(a.rows).glob("*.raw.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in need and all(isinstance(r.get(k), str) and r.get(k) for k in ("eng", "m_text", "hm", "ood")): raw[r["id"]] = r
    pool_by = collections.defaultdict(list); foreign = collections.defaultdict(list)
    for r in raw.values(): pool_by[r["lang"]].append(r)
    for L, v in pool_by.items():
        others = [x for L2, v2 in pool_by.items() if L2 != L for x in v2[:5]]
        foreign[L] = [w.strip(".,;:!?") for x in others[:40] for w in x["hm"].split() if len(w) >= 3][:600]
    ctx2 = {"pool_by_lang": pool_by, "lexicon": None, "foreign_other": foreign}; ids = [i for i in cand if i in raw]
    hm_rows, ood_rows = [], []
    for kind, ops, n, rows_out in (("hm", HM_OPS, a.hm_chains, hm_rows), ("ood", OOD_OPS, a.ood_chains, ood_rows)):
        made = 0
        while made < n and ids:
            r = raw[ids.pop()]; op, ch = chains_for(r, ops, rnd, ctx2)
            if not ch: st[kind + "_no_chain"] += 1; continue
            held = rnd.random() < a.heldout; made += 1
            for lv, row in [(0, r)] + [(c["level"], c["row"]) for c in ch]:
                x = copy.deepcopy(row); x["id"] = f"{r['id']}~{kind}~{op}~L{lv}"; x.update({"chain_base": r["id"], "chain_op": op if lv else "orig", "chain_level": lv, "heldout": held}); rows_out.append(x)
        st[kind + "_chains"] = made
    for name, rows_ in (("hm_rows", hm_rows), ("ood_rows", ood_rows)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for x in rows_: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        json.dump([x["id"] for x in rows_], open(out / f"{name}_ids.json", "w"))
    ops_used = collections.Counter(x["op"] for x in tr_items if x["level"] == 0)
    print(dict(st), "| translation items", len(tr_items), "hm items", len(hm_rows), "ood items", len(ood_rows), "| translation ops (chains):", dict(collections.Counter(x["op"] for x in tr_items if x["level"] == 1)))


if __name__ == "__main__":
    main()
