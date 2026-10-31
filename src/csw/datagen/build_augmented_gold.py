"""Pick the cleanest generated rows per language as candidate few-shots for the native speaker to correct.
python -m csw.datagen.build_augmented_gold [per_lang=2]  ->  exemplars/augmented_gold.jsonl  (same fields as gold.jsonl + provenance)"""
from csw.paths import DATA_ROOT
import json, re, glob, os, sys, collections as C
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import gates
from csw.datagen import cs_pipeline as pl
from csw.datagen.profiles import P, SUPPORT
per = int(sys.argv[1]) if len(sys.argv) > 1 else 2
vers = json.loads((HERE / "prompt_versions.json").read_text())
W = re.compile(r"[^\W\d_]+", re.U); tk = lambda t: [w.lower() for w in W.findall(t)]
GOLD_LANGS = {g["lang"] for g in json.loads("[" + ",".join(json.dumps(x) for x in __import__("cs_prompts").load_gold()) + "]")}
rows = []
for d in ("out_gb10", "out_gb10_t", "out_gb10_t2", "out_preview", "out_v3b"):
    for f in glob.glob(str(HERE / d / "*.raw.jsonl")):
        for l in open(f, encoding="utf-8").read().split("\n"):
            if l.strip():
                r = json.loads(l)
                if r.get("eng") and r.get("m_text") and r.get("hm") and r.get("ood"): r["_dir"] = d; rows.append(r)
def score(r):
    lang = r["lang"]; g = gates.check(dict(r), P[lang]); s = 0.0
    s += {"b2ab6a0c": 3, "c8c17d5c": 2}.get(r.get("prompt_fp"), 0)
    s -= 2 * len(g["warn"]) + 10 * len(g["fail"])
    e = set(tk(r["eng"])); h = tk(r["hm"]); share = sum(1 for w in h if w in e and w.isascii() and len(w) > 2) / max(1, len(h))
    if lang not in ("pcm", "gpe", "wes"): s -= abs(share - 0.16) * 10
    n = len(r["eng"].split()); s += 1 if 25 <= n <= 90 else -1
    if r["scenario"]["length"] == "long": s -= 3
    return s, g
by = C.defaultdict(list)
for r in rows:
    s, g = score(r); by[r["lang"]].append((s, r, g))
out = []; summary = []
for lang in P:
    if lang in GOLD_LANGS: continue
    cand = sorted(by.get(lang, []), key=lambda x: -x[0]); picked = []
    for s, r, g in cand:
        if all(r["scenario"]["genre"] != p[1]["scenario"]["genre"] for p in picked) or len(picked) == 0:
            picked.append((s, r, g))
        if len(picked) == per: break
    tier = next((t for t, v in SUPPORT.items() if lang in v), "unmeasured")
    for s, r, g in picked:
        out.append({"lang": lang, "genre": r["scenario"]["genre"], "switch": r["scenario"]["switch"], "ood_relation": r.get("ood_relation") or "idiomatic_paraphrase",
                    "eng": r["eng"], "m_text": r["m_text"], "hm": r["hm"], "ood": r["ood"], "devices": r.get("devices"),
                    "status": "CANDIDATE - native correction needed (delete this key when corrected)", "generator_support": tier,
                    "source_id": r["id"], "prompt_version": vers.get(r.get("prompt_fp") or "none", {}).get("version", "?"),
                    "auto_flags": g["fail"] + g["warn"], "dialect": r["scenario"]["dialect"]})
    summary.append((lang, len(picked), tier, [round(p[0], 1) for p in picked]))
(HERE / "exemplars" / "augmented_gold.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
print(len(out), "candidate rows for", len([s for s in summary if s[1]]), "languages ->", "exemplars/augmented_gold.jsonl")
print("no generated rows yet for:", [l for l in P if l not in by and l not in GOLD_LANGS])
print("only flagged rows available (score<0):", [s[0] for s in summary if s[1] and max(s[3]) < 0])
