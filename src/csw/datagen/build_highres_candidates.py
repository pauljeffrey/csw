"""Candidate few-shot rows for the HIGH-resource languages, for the native speaker to correct.
python -m csw.datagen.build_highres_candidates [target_own=5]  ->  exemplars/candidates_high_resource.jsonl
Each language is topped up to `target_own` corrected examples (need = target - corrected rows already in the gold files, +1 spare). Candidates are the cleanest
generated rows by the rule-based scorer (formal eng, hm share in band, diacritics, no gate failure), varied in genre and pinned switch language.
Rows keep a `status` key, so the prompt loader IGNORES them until you correct the text, delete `status`, and move the row into augmented_gold.jsonl."""
from csw.paths import DATA_ROOT
import json, sys, glob, os, collections as C
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import score_rows as sr
from csw.datagen.profiles import HIGH, GOLD_N, P
target = int(sys.argv[1]) if len(sys.argv) > 1 else 5
rows = C.defaultdict(list)
for d in ("out_gb10", "out_gb10_t", "out_gb10_t2", "out_preview"):
    for f in glob.glob(str(HERE / d / "*.raw.jsonl")):
        lang = os.path.basename(f).split(".")[0]
        if lang not in HIGH: continue
        for l in open(f, encoding="utf-8").read().split("\n"):
            if not l.strip(): continue
            r = json.loads(l)
            if not all(r.get(k) for k in ("eng", "m_text", "hm", "ood")) or r["scenario"]["length"] == "long" or not (25 <= len(r["eng"].split()) <= 85): continue
            s = sr.score_row(r)
            if s["flags"] or (s["components"].get("eng_formal") or 0) < 0.95: continue
            rows[lang].append((s["rule_score"], r))
out, summary = [], {}
for lang in HIGH:
    need = max(0, target - GOLD_N.get(lang, 0)); want = need + 1 if need else 0
    picked, seen_g, seen_s = [], set(), set()
    for sc, r in sorted(rows[lang], key=lambda x: -x[0]):
        g, sw = r["scenario"]["genre"], r["scenario"]["switch"]
        if len(picked) < want and (g not in seen_g or len(seen_g) >= want) and (sw not in seen_s or len(seen_s) >= len(P[lang].switch)):
            picked.append((sc, r)); seen_g.add(g); seen_s.add(sw)
    for sc, r in picked:
        out.append({"lang": lang, "genre": r["scenario"]["genre"], "switch": r["scenario"]["switch"], "ood_relation": r.get("ood_relation") or "idiomatic_paraphrase",
                    "eng": r["eng"], "m_text": r["m_text"], "hm": r["hm"], "ood": r["ood"], "devices": r.get("devices"), "dialect": r["scenario"]["dialect"],
                    "status": "CANDIDATE - correct the text, delete this key, then move the row to augmented_gold.jsonl"})
    summary[lang] = (GOLD_N.get(lang, 0), need, len(picked), len(rows[lang]))
(HERE / "exemplars" / "candidates_high_resource.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
print(f"{len(out)} candidate rows for {sum(1 for v in summary.values() if v[2])} high-resource languages -> exemplars/candidates_high_resource.jsonl")
print("lang: corrected-already / needed / candidates given / clean rows available:")
for l, v in summary.items(): print(f"  {l}: {v[0]} / {v[1]} / {v[2]} / {v[3]}")
