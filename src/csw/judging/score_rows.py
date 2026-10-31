"""Rule-based quality scores for code-switching rows (no model involved). Every component is in [0,1]; `rule_score` is their weighted
mean multiplied by hard-failure gates. The formulas are documented in DATASET_CARD.md (section "Rule-based scores").
  python -m csw.judging.score_rows <out_dir> [lang ...]      -> <lang>.scored.jsonl + scores_summary.json/.md
  python -m csw.judging.score_rows --gold                    -> scores the human gold rows (sanity check: they should score high)"""
from csw.paths import DATA_ROOT
import difflib, json, re, sys, glob, os, statistics as st, collections as C
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import gates
from csw.datagen.profiles import P
from csw.datagen import cs_prompts as pr
W = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.U); tk = lambda t: [w.lower() for w in W.findall(t or "")]
EMOTIVE = set("heartbreaking heartbroken tragedy tragic terrible terribly awful horrible horrific wonderful amazing incredible unbearable devastating shocking disaster nightmare miracle miraculous thrilled furious outrageous disgusting beautiful gorgeous magnificent fantastic brilliant dreadful appalling excruciating overjoyed ecstatic miserable agonizing painful scary frightening heavenly blessed cursed".split())
SLANG = set("gonna wanna gotta kinda sorta yeah yep nope hey wow oh ooh ugh hmm lol dude buddy guys stuff things gotten ain't".split())
FIGURE = re.compile(r"\b(like a|as if|as though|as \w+ as a|piece of cake|heart of|break the ice|raining cats|under the weather|hit the road|a drop in the ocean|tip of the iceberg|roots?\b.*\btree|sold (his|her|their) soul|traded (his|her|their) soul)\b", re.I)
CONTR = re.compile(r"\b\w+(n't|'re|'ll|'ve|'m|'d)\b|\b(it|he|she|that|there|what|who)'s\b", re.I)
NEED = {"yor": r"[ẹọṣàáèéìíòóùúńǹ]", "ibo": r"[ịọụṅ]", "ewe": r"[ɖɛƒɣɔŋʋ]", "twi": r"[ɛɔ]", "hau": r"[ɓɗƙƴ]", "fon": r"[ɖɔɛ]", "wol": r"[ñŋëóàé]", "lin": r"[ɛɔáéíóú]", "bam": r"[ɛɔɲŋ]", "ful": r"[ɓɗŋɲƴ]", "kik": r"[ĩũ]", "mos": r"[ɛɩʋɔãẽĩõũ]", "bin": r"[ẹọ]", "efi": r"[ọụịń]", "ibb": r"[ọụịṅḿń]"}
def clip(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))
def band(x, lo, hi, slack):          # 1 inside [lo,hi]; linear fall to 0 over `slack` outside
    return 1.0 if lo <= x <= hi else clip(1 - ((lo - x) if x < lo else (x - hi)) / slack)
def dist3(ws): 
    g = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]; return len(set(g)) / max(1, len(g))
def loop(ws):
    for n in (1, 2, 3, 4):
        for i in range(len(ws) - n * 4 + 1):
            if all(ws[i + k * n:i + (k + 1) * n] == ws[i:i + n] for k in range(1, 4)): return True
    return False
_vocab, _seeds, _ratio = {}, {}, {}
def vocab(lang):
    if lang in _vocab: return _vocab[lang]
    docs = []
    for name in (f"{lang}.vast.jsonl", f"{lang}.jsonl"):
        p = HERE / "seeds" / name
        if p.exists(): docs += [json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()]
    for r in gold_rows(lang): docs.append({"text": " ".join(r.get(k, "") for k in ("m_text", "ood")), "seed_id": ""})
    _seeds[lang] = {d.get("seed_id"): d["text"] for d in docs if d.get("seed_id")}
    _vocab[lang] = set(w for d in docs for w in tk(d["text"])) if len(docs) >= 40 else None
    return _vocab[lang]
_gold = None
def gold_rows(lang=None):
    global _gold
    if _gold is None:
        _gold = [dict(g, human_gold=True) for g in pr.load_gold()] + [dict(r, human_gold=True) for v in pr.load_aug_gold().values() for r in v]
    return [g for g in _gold if lang is None or g["lang"] == lang]
def expected_ratio(lang):           # words(m_text)/words(eng): measured on the gold rows of the language, else a generic band centre
    if lang not in _ratio:
        r = [len(tk(g["m_text"])) / max(1, len(tk(g["eng"]))) for g in gold_rows(lang)]
        _ratio[lang] = st.median(r) if len(r) >= 2 else None
    return _ratio[lang]
def oov(lang, text):
    v = vocab(lang)
    if not v: return None
    ws = [w for w in tk(text) if len(w) > 2]
    return sum(1 for w in ws if w not in v) / max(1, len(ws)) if ws else None
def score_row(r):
    lang = r["lang"]; p = P[lang]; s = r.get("scenario", {}); sw = s.get("switch") or r.get("switch") or ""
    eng, m, hm, ood = (r.get(k) or "" for k in ("eng", "m_text", "hm", "ood")); S = {}; flags = []
    ew, mw, hw, ow = tk(eng), tk(m), tk(hm), tk(ood)
    # 1 structure
    S["structure"] = sum(bool(x.strip()) for x in (eng, m, hm, ood)) / 4
    # 2 eng formality: emotive words, slang, figures, contractions, exclamations
    emo = sum(1 for w in ew if w in EMOTIVE); sl = sum(1 for w in ew if w in SLANG); fig = len(FIGURE.findall(eng)); con = len(CONTR.findall(eng)); exc = eng.count("!")
    S["eng_formal"] = clip(1 - 0.25 * emo - 0.2 * sl - 0.25 * fig - 0.1 * con - 0.15 * exc)
    if emo or fig: flags.append("eng_expressive")
    # 3 m_text length vs expected ratio for the language
    ratio = len(mw) / max(1, len(ew)); exp = expected_ratio(lang) or 1.0
    S["m_text_length"] = band(ratio / exp, 0.7, 1.4, 0.8)
    # 4 m_text lexical naturalness: invented-word proxy (words absent from real text of the language), relative to real text
    om = oov(lang, m); S["m_text_lexicon"] = None if om is None else clip(1 - max(0.0, om - 0.12) / 0.35)
    # 5 hm mix: share of switch-language words in the target band 8-30%
    e = set(ew)
    if sw == "English": share = sum(1 for w in hw if w in e and w.isascii() and len(w) > 2) / max(1, len(hw))
    else:
        oh, om2 = oov(lang, hm), oov(lang, m); share = None if oh is None or om2 is None else max(0.0, oh - om2)
    S["hm_mix"] = None if share is None else (band(share, 0.12, 0.42, 0.20) if sw == "English" else band(share, 0.08, 0.50, 0.25))   # target 15-40%: the model overshoots, so the band stops at ~40% (gold median 36%, p90 62%)
    if share is not None and share > (0.55 if sw == "English" else 0.70): flags.append("hm_over_mixed")
    # 6 hm differs from m_text (a spoken variation, not a copy and not a rewrite)
    sim = difflib.SequenceMatcher(None, mw, hw).ratio() if mw and hw else 0
    S["hm_vs_m_text"] = 0.0 if hm.strip() == m.strip() else (band(sim, 0.5, 0.92, 0.3) if lang in ("pcm", "gpe", "wes") else band(sim, 0.07, 0.75, 0.3))
    # 7 ood richness: longer/richer than m_text, varied vocabulary, devices listed, not a copy
    rr = len(ow) / max(1, len(mw)); ttr = len(set(ow)) / max(1, len(ow)); nd = len(r.get("devices") or [])
    S["ood_richness"] = 0.0 if ood.strip() == m.strip() else (0.4 * band(rr, 1.05, 2.0, 0.8) + 0.3 * clip((ttr - 0.45) / 0.25) + 0.3 * clip(nd / 3))
    # 8 diacritics: required marks present in hm and ood (and in the pinned language when it needs them)
    need = NEED.get(lang)
    def dens(t): 
        L = sum(c.isalpha() for c in t); return len(re.findall(need, t, re.I)) / L if L else 0
    S["diacritics"] = 1.0 if not need else clip(0.5 * min(1, dens(hm) / 0.03) + 0.5 * min(1, dens(ood) / 0.03))
    # 9 repetition / loops
    S["repetition"] = 0.0 if any(loop(x) for x in (mw, hw, ow)) else clip(min(dist3(x) for x in (mw, hw, ow) if len(x) > 8) if any(len(x) > 8 for x in (mw, hw, ow)) else 1.0)
    # 10 pinned-language discipline + Pidgin purity (from the deterministic gates)
    g = gates.check({"eng": eng, "m_text": m, "hm": hm, "ood": ood, "scenario": {"switch": sw, **{k: (r.get("scenario") or {}).get(k) for k in ("length_spec", "n_sent")}}}, p)
    pin = 1.0 - (0.5 if "english_when_not_pinned" in g["warn"] else 0) - (0.3 if "pidgin_leak_in_hm" in g["warn"] else 0)
    pidgin = 1.0 if not (lang in ("pcm", "gpe", "wes") or "Pidgin" in sw) else clip(1 - 0.3 * g["metrics"].get("pidgin_formal", 0))
    S["pinned_language"] = clip(pin); S["pidgin_purity"] = pidgin
    # 11 copy of the seed (7-word overlap)
    S["seed_copy"] = 1.0
    sid = r.get("seed_id"); vocab(lang)
    if sid and _seeds.get(lang, {}).get(sid):
        sd = tk(_seeds[lang][sid]); grams = {tuple(sd[i:i + 7]) for i in range(len(sd) - 6)}
        if any(tuple(x[i:i + 7]) in grams for x in (mw, hw, ow) for i in range(len(x) - 6)): S["seed_copy"] = 0.0; flags.append("copies_seed")
    for f in g["fail"]: flags.append("gate:" + f)
    W = {"eng_formal": .20, "m_text_lexicon": .15, "hm_mix": .15, "hm_vs_m_text": .10, "ood_richness": .10, "diacritics": .10, "m_text_length": .05, "pinned_language": .05, "repetition": .05, "pidgin_purity": .05}
    num = sum(W[k] * S[k] for k in W if S.get(k) is not None); den = sum(W[k] for k in W if S.get(k) is not None)
    mult = (1.0 if S["structure"] == 1 else 0.0) * S["seed_copy"] * (0.0 if any(f.startswith("gate:") for f in flags) else 1.0)
    S = {k: (None if v is None else round(v, 3)) for k, v in S.items()}
    diff = r.get("difficulty")
    return {"components": S, "rule_score": round(mult * num / den if den else 0.0, 3), "flags": flags, "model_difficulty": diff if isinstance(diff, (int, float)) else None}
def main():
    if "--gold" in sys.argv:
        rows = gold_rows(); res = [score_row(dict(g, scenario={"switch": g.get("switch")})) for g in rows]
        by = C.defaultdict(list)
        for g, x in zip(rows, res): by[g["lang"]].append(x["rule_score"])
        print(f"human gold rows scored: {len(rows)} | mean rule_score {st.mean(x['rule_score'] for x in res):.3f} | by flag:", C.Counter(f for x in res for f in x["flags"]).most_common(6))
        print("lowest gold rows:", sorted(((x["rule_score"], g["lang"]) for g, x in zip(rows, res)))[:6]); return
    od = Path(sys.argv[1]); langs = sys.argv[2:] or sorted({os.path.basename(f).split(".")[0] for f in glob.glob(str(od / "*.raw.jsonl"))})
    summ = {}
    for lang in langs:
        if lang not in P: continue                       # retired languages (e.g. urh) are left unscored
        rows = [json.loads(x) for x in (od / f"{lang}.raw.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
        out = []
        for r in rows:
            if not all(r.get(k) for k in ("eng", "m_text", "hm", "ood")): continue
            sc = score_row(r); out.append(dict(r, scores=sc))
        (od / f"{lang}.scored.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
        comp = {k: [x["scores"]["components"][k] for x in out if x["scores"]["components"].get(k) is not None] for k in (out[0]["scores"]["components"] if out else [])}
        dif = [(x["scores"]["model_difficulty"], x["scores"]["rule_score"]) for x in out if x["scores"].get("model_difficulty") is not None]
        corr = None
        if len(dif) > 8 and len({d for d, _ in dif}) > 1:
            mx, my = st.mean(d for d, _ in dif), st.mean(s_ for _, s_ in dif); sx, sy = st.pstdev([d for d, _ in dif]), st.pstdev([s_ for _, s_ in dif])
            corr = round(sum((d - mx) * (s_ - my) for d, s_ in dif) / (len(dif) * sx * sy), 3) if sx and sy else None
        summ[lang] = {"rows": len(out), "rule_score_mean": round(st.mean(x["scores"]["rule_score"] for x in out), 3) if out else None, "components": {k: round(st.mean(v), 3) for k, v in comp.items() if v},
                      "difficulty_mean": round(st.mean(d for d, _ in dif), 2) if dif else None, "difficulty_vs_rule_score_corr": corr}
    json.dump(summ, open(od / "scores_summary.json", "w"), indent=1)
    allm = [v["rule_score_mean"] for v in summ.values() if v["rule_score_mean"] is not None]
    print(f"scored {sum(v['rows'] for v in summ.values())} rows in {len(summ)} languages; mean of language means {st.mean(allm):.3f}")
    for l, v in sorted(summ.items(), key=lambda kv: -(kv[1]["rule_score_mean"] or 0))[:6]: print(f"  best  {l} {v['rule_score_mean']}")
    for l, v in sorted(summ.items(), key=lambda kv: (kv[1]["rule_score_mean"] or 0))[:6]: print(f"  worst {l} {v['rule_score_mean']}")
if __name__ == "__main__": main()
