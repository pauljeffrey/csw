"""Assemble the dataset: every human-gold row + every generated row, each with rule-based scores and tags.   python -m csw.datagen.assemble_final [out_dir ...]
Output: final/<lang>.jsonl, final/analytics.json, final/analytics.md.  Nothing is dropped: filter on `source`, `prompt_version`, `scores.rule_score`, `difficulty`."""
from csw.paths import DATA_ROOT
import json, sys, glob, os, statistics as st, collections as C
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import score_rows as sr
from csw.datagen import cs_prompts as pr
from csw.datagen.profiles import P, TARGETS
dirs = [a for a in sys.argv[1:] if not a.startswith("--")] or ["out_gb10", "out_gb10_t", "out_gb10_t2", "out_preview", "out_main"]
vers = json.loads((HERE / "prompt_versions.json").read_text()); cur = max((v["version"] for v in vers.values()), key=lambda s: int(s.split(".")[1]))
out = HERE / "final"; out.mkdir(exist_ok=True); rows = C.defaultdict(list)
def tags_of(r, sc): return list(sc["flags"])
gold = [dict(g, _src="gold.jsonl") for g in pr.load_gold()] + [dict(r, _src="augmented_gold.jsonl") for v in pr.load_aug_gold().values() for r in v]
for n, g in enumerate(gold):
    if g["lang"] not in P: continue
    sc = sr.score_row(dict(g, scenario={"switch": g.get("switch")}))
    rows[g["lang"]].append({"id": f"gold-{g['lang']}-{n:04d}", "lang": g["lang"], "source": "human_gold", "prompt_version": None, "eng_initial": None, "eng": g["eng"], "m_text": g["m_text"], "hm": g["hm"], "ood": g["ood"],
                            "ood_relation": g.get("ood_relation"), "devices": g.get("devices"), "scenario": {"switch": g.get("switch"), "genre": g.get("genre"), "dialect": g.get("dialect"), "meta_note": "genre/switch/dialect on gold rows are unverified hints"},
                            "difficulty": None, "scores": sc, "quality_tags": tags_of(g, sc), "provenance": g["_src"]})
seen = set()
for d in dirs:
    for f in sorted(glob.glob(str(HERE / d / "*.raw.jsonl"))):
        lang = os.path.basename(f).split(".")[0]
        if lang not in P: continue
        for l in open(f, encoding="utf-8").read().split("\n"):
            if not l.strip(): continue
            r = json.loads(l)
            if r["id"] in seen or not all(r.get(k) for k in ("eng", "m_text", "hm", "ood")): continue
            seen.add(r["id"]); sc = sr.score_row(r); v = vers.get(r.get("prompt_fp") or "none", {"version": "v3.?"})["version"]
            rows[lang].append({"id": r["id"], "lang": lang, "source": "generated_given_eng" if r.get("source_row") else "generated", "prompt_version": v, "superseded": v != cur,
                               "eng_initial": r.get("eng_initial"), "eng": r["eng"], "m_text": r["m_text"], "hm": r["hm"], "ood": r["ood"], "ood_relation": r.get("ood_relation"), "devices": r.get("devices"),
                               "scenario": r["scenario"], "anchors": r.get("anchors"), "seed_id": r.get("seed_id"), "seed_src": r.get("seed_src"), "source_row": r.get("source_row") or None,
                               "conf": r.get("conf"), "difficulty": r.get("difficulty"), "self_check": r.get("self_check"), "flags": r.get("flags"), "teacher": r.get("teacher"), "scores": sc, "quality_tags": tags_of(r, sc)})
an = {}
for lang, rs in rows.items():
    (out / f"{lang}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rs) + "\n", encoding="utf-8")
    g = [x for x in rs if x["source"] == "human_gold"]; ge = [x for x in rs if x["source"] != "human_gold"]
    m = lambda L: round(st.mean(x["scores"]["rule_score"] for x in L), 3) if L else None
    an[lang] = {"target": TARGETS.get(lang), "gold": len(g), "generated": len(ge), "rule_score_gold": m(g), "rule_score_generated": m(ge),
                "current_version_rows": sum(1 for x in ge if not x.get("superseded")), "hard_gate_failures": sum(1 for x in ge if any(t.startswith("gate:") for t in x["quality_tags"]))}
json.dump(an, open(out / "analytics.json", "w"), indent=1)
lines = ["| lang | target | gold | generated | rule_score gold | rule_score generated | current-prompt rows | hard-gate failures |", "|---|---|---|---|---|---|---|---|"]
for l, v in sorted(an.items(), key=lambda kv: -(kv[1]["target"] or 0)): lines.append(f"| {l} | {v['target']} | {v['gold']} | {v['generated']} | {v['rule_score_gold']} | {v['rule_score_generated']} | {v['current_version_rows']} | {v['hard_gate_failures']} |")
(out / "analytics.md").write_text("\n".join(lines), encoding="utf-8")
tg = sum(v["gold"] for v in an.values()); tr = sum(v["generated"] for v in an.values())
print(f"{tg} gold rows + {tr} generated rows in {len(an)} languages -> final/ ; current prompt version {cur}")
print("gold mean rule_score", round(st.mean(v["rule_score_gold"] for v in an.values() if v["rule_score_gold"] is not None), 3), "| generated mean", round(st.mean(v["rule_score_generated"] for v in an.values() if v["rule_score_generated"] is not None), 3))
