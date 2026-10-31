"""SFT set for csw: five tasks (eng->hm, m->hm, eng->ood, m->ood, hm->ood) from the generated rows (repaired -hm2 rows replace their leaky parents), plus
  * the LLM judges' corrected copies of rows (csw.sft.judge_fixes: -fxf / -fxl rows) next to the originals; a field a judge called major/critical is withheld from the original,
  * replay of eng->m from HUMAN-translated pairs (--replay file.jsonl {lang, eng, m}); the generated m_text is not used for replay,
  * a curated test split: rows the judges found error-free and top-graded (--eval-per-lang); every other row sharing their English is dropped from SFT; the old random 2% 'test' bucket is test_rand.
python -m csw.sft.build_sft2 --rows work/post_rows --out work/csw_sft [--lexicon work/lexicon.json] [--replay work/replay.jsonl] [--no-fixes]
Split by sha1(english): 2% val, rest train (plus the curated test).  Gold rows are train-only."""
from __future__ import annotations
from csw.paths import DATA_ROOT
import argparse, collections, glob, hashlib, json, random, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import gates                                   # noqa: E402
from csw.datagen.profiles import P                         # noqa: E402
from csw.sft import fmt   # noqa: E402
from csw.sft import devices as dv   # noqa: E402
from csw.sft import build_sft as B   # noqa: E402
from csw.sft import fmt2       # noqa: E402
from csw.sft import judge_fixes as JF       # noqa: E402

EVAL_ROWS, EVAL_SHAS = set(), set()                          # curated test: row ids / sha1 of their English


def eng_sha(eng): return hashlib.sha1(eng.strip().lower().encode()).hexdigest()


def split_of(r):
    if r["source"] == "gold": return "train"
    if r["id"] in EVAL_ROWS: return "test"
    if eng_sha(r["eng"]) in EVAL_SHAS: return "drop"
    s = B.split_of(r["eng"], r["source"])
    return "test_rand" if s == "test" else s

DROP_WARN_HM = B.DROP_WARN_HM


def examples(r, lex, st):
    lang = r["lang"]
    if lang not in fmt.NAMES or lang not in P:
        st["skip_lang"] += 1; return []
    for k in ("eng", "m_text", "hm", "ood"):
        r[k] = B.clean(r[k])
        if len(r[k]) > B.MAX_CHARS: st["skip_too_long"] += 1; return []
    g = {"fail": [], "warn": []}
    if r["source"] != "gold" and (r.get("scenario") or {}).get("length_spec"):
        g = gates.check(r, P[lang])
    fails = [f for f in g["fail"] if f != "english_leak_unpinned"]
    if fails:
        st["row_dropped_gate_fail"] += 1; return []
    leak = "english_leak_unpinned" in g["fail"] or (r["source"] != "gold" and bool(gates.en_leak(r) and gates.en_leak(r)[0] >= 2 and gates.en_leak(r)[1] > gates.EN_LEAK_MAX_SHARE and r["lang"] not in ("pcm", "gpe", "wes") and (r.get("scenario") or {}).get("switch") not in ("English", "Sheng", "Nigerian Pidgin", "Ghanaian Pidgin")))
    sw = fmt.SWITCH_TAG.get((r.get("scenario") or {}).get("switch"))
    if sw is None: st["skip_switch"] += 1; return []
    split = split_of(r); out = []
    if split == "drop": st["row_dropped_eval_english"] += 1; return []
    bad = set(r.get("bad_fields") or [])

    def add(task, prm, comp, extra=None):
        out.append({"id": f"{r['id']}:{task}", "lang": lang, "task": task, "split": split, "source": r["source"], "prompt": prm, "completion": comp, "switch": sw, "meta": extra or {}})
        st["ex:" + task] += 1
    h = dv.hm_devices(r, lex); hm_ok = False
    if leak: st["hm_dropped_en_leak"] += 1
    elif "hm" in bad: st["hm_dropped_judge_flag"] += 1
    elif not h["devices"]: st["hm_dropped_no_switch_found"] += 1
    elif not 0.04 <= h["switch_ratio"] <= 0.65: st["hm_dropped_ratio"] += 1
    elif set(g["warn"]) & DROP_WARN_HM: st["hm_dropped_warn"] += 1
    else: hm_ok = True
    if hm_ok:
        comp = fmt.completion_hm(lang, sw, h["devices"], r["hm"])
        add("eng_hm", fmt2.prompt("eng_hm", lang, sw, r["eng"]), comp, {"devices": h["devices"]})
        if "m_text" not in bad: add("m_hm", fmt2.prompt("m_hm", lang, sw, r["m_text"]), comp, {"devices": h["devices"]})
        else: st["m_src_dropped_judge_flag"] += 1
    o = dv.ood_devices(r)
    if len(o["devices"]) < 2: st["ood_dropped_few_devices"] += 1
    elif "ood" in bad: st["ood_dropped_judge_flag"] += 1
    else:
        osw = dv.hm_devices(r, lex, field="ood")["switch_ratio"]; tag2 = sw if osw >= B.OOD_SWITCH_MIN else None
        comp = fmt.completion_ood(lang, o["devices"], r["ood"], tag2)
        add("eng_ood", fmt2.prompt("eng_ood", lang, None, r["eng"]), comp, {"devices": o["devices"]})
        if "m_text" not in bad: add("m_ood", fmt2.prompt("m_ood", lang, None, r["m_text"]), comp, {"devices": o["devices"]})
        else: st["m_src_dropped_judge_flag"] += 1
        if hm_ok and "hm" not in bad: add("hm_ood", fmt2.prompt("hm_ood", lang, sw, r["hm"]), comp, {"devices": o["devices"]})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); ap.add_argument("--lexicon", default="work/lexicon.json")
    ap.add_argument("--gold", nargs="*", default=[str(HERE / "exemplars" / "augmented_gold.jsonl")]); ap.add_argument("--pidgin-cap", type=int, default=9000)
    ap.add_argument("--replay", default=""); ap.add_argument("--no-fixes", action="store_true"); ap.add_argument("--eval-per-lang", type=int, default=0); ap.add_argument("--relax-minor", action="store_true"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = B.load_rows(a.rows, [g for g in a.gold if Path(g).exists()])
    st = collections.Counter(rows_in=len(rows))
    variants = []
    if not a.no_fixes:
        runs = JF.load(HERE)
        variants, bad = JF.variants_and_bad(runs)
        for r in rows:
            if r["id"] in bad: r["bad_fields"] = sorted(bad[r["id"]]); st["rows_with_judge_flag"] += 1
        if a.eval_per_lang:
            ev = JF.eval_ids(runs, a.eval_per_lang, a.relax_minor)
            ids = {i for v in ev.values() for i in v}; present = {r["id"]: r for r in rows}
            EVAL_ROWS.update(i for i in ids if i in present); EVAL_SHAS.update(eng_sha(present[i]["eng"]) for i in EVAL_ROWS)
            st["eval_rows"] = len(EVAL_ROWS); (out / "eval_ids.json").write_text(json.dumps({l: [i for i in v if i in present] for l, v in ev.items()}))
    rows.sort(key=lambda r: 0 if r.get("parent_id") else 1)            # repaired variants first: they win the (lang, eng, m_text) dedupe over their leaky parents
    seen, uniq = set(), []
    for r in rows:
        k = (r["lang"], r["eng"].strip().lower(), r["m_text"].strip().lower())
        if k not in seen: seen.add(k); uniq.append(r)
    full = lambda r: (r["lang"], r["eng"].strip().lower(), r["m_text"].strip().lower(), r["hm"].strip().lower(), r["ood"].strip().lower())
    seen_full = {full(r) for r in uniq}                                # corrected copies sit next to the original unless they came out identical
    for v in variants:
        if full(v) not in seen_full: seen_full.add(full(v)); uniq.append(v); st["judge_fix_rows"] += 1
    st["rows_after_dedupe"] = len(uniq)
    lex = dv.Lexicon.load(a.lexicon); allx = []
    for r in uniq: allx += examples(r, lex, st)
    if a.replay:                                                       # human-translated eng->m pairs: {"lang","eng","m","source"}
        for l in open(a.replay, encoding="utf-8"):
            if not l.strip(): continue
            x = json.loads(l)
            if x["lang"] not in fmt.NAMES: continue
            sp = "train" if eng_sha(x["eng"]) not in EVAL_SHAS and B.split_of(x["eng"], "x") == "train" else ("val" if B.split_of(x["eng"], "x") == "val" else None)
            if sp is None: continue
            allx.append({"id": f"replay-{eng_sha(x['eng'])[:10]}-{x['lang']}", "lang": x["lang"], "task": "eng_m", "split": sp, "source": x.get("source", "human"),
                         "prompt": fmt2.prompt("eng_m", x["lang"], None, B.clean(x["eng"])), "completion": fmt.completion_m_text(x["lang"], B.clean(x["m"])), "switch": None, "meta": {}})
            st["ex:eng_m"] += 1
    seen_ex, dd = set(), []                                             # a corrected copy can reproduce an original example exactly
    for e in allx:
        k = (e["prompt"], e["completion"])
        if k not in seen_ex: seen_ex.add(k); dd.append(e)
    st["examples_dedupe_removed"] = len(allx) - len(dd); allx = dd
    rnd = random.Random(0); by = collections.defaultdict(list)
    for e in allx: by[(e["lang"], e["task"], e["split"])].append(e)
    final = []
    for (lang, task, split), v in by.items():
        if split == "train" and lang in ("pcm", "gpe", "wes") and len(v) > a.pidgin_cap:
            rnd.shuffle(v); v = v[: a.pidgin_cap]
        final += v
    rnd.shuffle(final)
    for split in ("train", "val", "test", "test_rand"):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for e in final:
                if e["split"] == split: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    cnt = collections.Counter((e["split"], e["task"]) for e in final)
    (out / "stats.json").write_text(json.dumps({"counters": dict(st), "by_split_task": {f"{s}/{t}": n for (s, t), n in sorted(cnt.items())}}, indent=1))
    print(json.dumps({"rows": st["rows_after_dedupe"], **{f"{s}/{t}": n for (s, t), n in sorted(cnt.items())}}, indent=0)); print({k: v for k, v in st.items() if k.startswith(("hm_", "ood_", "skip", "row_"))})


if __name__ == "__main__":
    main()
