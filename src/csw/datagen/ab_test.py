"""A/B test of prompt variants on cards that were ALREADY generated (same ids, same cards). Run on the GPU box with the main run paused.
CS_RULES_FIRST=1 CS_TOTAL_SHOTS=5 python -m csw.datagen.ab_test --tag B [--per-lang 6]   ->  out_ab/<tag>.jsonl + summary (wall time, rows/min, tokens/s, rule score, gate failures, length adherence)
The control (tag A) is the production prompt (7 shots, rules last) on the same cards, so speed and quality are compared under identical load."""
from csw.paths import DATA_ROOT
import json, os, sys, time, glob, random, statistics as st, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = DATA_ROOT
os.environ.setdefault("CS_OUT", "out_ab")
from csw.datagen import cs_pipeline as pl
from csw.datagen import cs_prompts as pr
from csw.judging import score_rows as sr
from csw.datagen.providers import Request
from csw.datagen.profiles import TARGETS, LENGTHS, SUPPORT, HOLD
HOLD_L = set(HOLD)
a = sys.argv; opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
tag, per = opt("--tag", "A"), int(opt("--per-lang", 6)); model = "google/gemma-4-31b-it"
pl.PROVIDER = "local"; pl.GOLD = pr.load_gold()
rnd = random.Random(11); done = collections.defaultdict(list)
for f in glob.glob(str(HERE / "out_main" / "*.raw.jsonl")):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if all(r.get(k) for k in ("eng", "m_text", "hm", "ood")) and not r.get("source_row") and (r["lang"] in SUPPORT["ok"] or (os.environ.get("AB_ALL") == "1" and r["lang"] in TARGETS and r["lang"] not in HOLD_L)): done[r["lang"]].append(r)
ids = []
for lang, rs in sorted(done.items()):
    rnd.shuffle(rs); seen = set(); pick = []
    for r in rs:                                           # spread over lengths
        k = r["scenario"]["length"]
        if k not in seen or len(pick) >= 3: pick.append(r); seen.add(k)
        if len(pick) >= per: break
    ids += [(lang, r["id"], r) for r in pick]
cards = []
for lang in sorted({l for l, _, _ in ids}):
    want = {i for l, i, _ in ids if l == lang}
    cards += [c for c in pl.make_plan(lang, TARGETS[lang]) if c["id"] in want]
orig = {r["id"]: r for _, _, r in ids}
print(f"[{tag}] variant={pr.VARIANT_TAG or 'production'} shots={pr.TOTAL_SHOTS} rules_first={pr.RULES_FIRST} | {len(cards)} cards, {len({c['lang'] for c in cards})} languages", flush=True)
CONC = int(os.environ.get("AB_CONC", 48)); pv = pl.prov(model, CONC); out = []
def one(c):
    t0 = time.time()
    r = pv.complete(Request(c["id"], pr.gen_prompt(pl.P[c["lang"]], c, pl.GOLD), max_tokens=LENGTHS[c["length"]][2], temperature=0.6, top_p=0.95, response_format={"type": "json_object"}))
    d = pr.parse_json(r.text) if r.ok else None
    if not d or not all(d.get(k) for k in ("eng", "m_text", "hm", "ood")): return {"id": c["id"], "lang": c["lang"], "ok": False, "usage": [r.prompt_tokens, r.completion_tokens], "sec": time.time() - t0}
    row = {"id": c["id"], "lang": c["lang"], "scenario": {"switch": c["switch"], "length": c["length"], "n_sent": c["n_sent"], "length_spec": "v2", "genre": c["genre"], "track": c["track"], "dialect": c["dialect"]}, "seed_id": c["seed_id"],
           **{k: d.get(k) for k in ("eng_initial", "eng", "m_text", "hm", "ood", "difficulty")}}
    s = sr.score_row(row); g = sr.gates.check(dict(row), pl.P[c["lang"]]); en = sr.gates.sentences(row["eng"]); want = c["n_sent"]; tol = 1 if want <= 6 else (2 if want <= 12 else 3)
    return {"id": c["id"], "lang": c["lang"], "ok": True, "row": row, "score": s["rule_score"], "flags": s["flags"], "hard": any(f.startswith("gate:") for f in s["flags"]), "len_ok": abs(en - want) <= tol,
            "len_exact": en == want, "hm_en": g["metrics"].get("en_hm"), "usage": [r.prompt_tokens, r.completion_tokens], "sec": time.time() - t0, "orig_score": sr.score_row(orig[c["id"]])["rule_score"]}
t0 = time.time()
with ThreadPoolExecutor(CONC) as ex: out = list(ex.map(one, cards))
wall = time.time() - t0; ok = [o for o in out if o["ok"]]
Path(HERE / "out_ab").mkdir(exist_ok=True)
(HERE / "out_ab" / f"{tag}.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
ct = sum(o["usage"][1] for o in out); pt = sum(o["usage"][0] for o in out)
summ = {"tag": tag, "variant": pr.VARIANT_TAG or "production", "cards": len(out), "valid": len(ok), "wall_s": round(wall), "rows_per_hour": round(len(ok) / wall * 3600), "completion_tok_per_s": round(ct / wall), "prompt_tokens_mean": round(pt / len(out)),
        "rule_score": round(st.mean(o["score"] for o in ok), 3), "orig_rule_score_same_cards": round(st.mean(o["orig_score"] for o in ok), 3), "hard_fail_rate": round(sum(o["hard"] for o in ok) / len(ok), 3),
        "length_ok": round(sum(o["len_ok"] for o in ok) / len(ok), 3), "length_exact": round(sum(o["len_exact"] for o in ok) / len(ok), 3), "mean_difficulty": round(st.mean(o["row"]["difficulty"] for o in ok if isinstance(o["row"].get("difficulty"), (int, float))), 2)}
(HERE / "out_ab" / f"{tag}.summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ), flush=True)
