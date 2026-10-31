"""Track D pool: the formal `eng` of already generated rows becomes the given English for other languages.
python -m csw.datagen.build_eng_pool <out_dir> [<out_dir> ...]   -> eng_pool.jsonl
Keeps rows of the CURRENT prompt version (or all with --all) that have no hard gate failure; one record per distinct eng."""
from csw.paths import DATA_ROOT
import json, re, sys, glob, os, hashlib
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import gates
from csw.datagen import cs_pipeline as pl
from csw.datagen.profiles import REUSE_FROM, REUSE_DEFAULT_SOURCES
allv = "--all" in sys.argv; dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
vers = json.loads((HERE / "prompt_versions.json").read_text()); cur = {k for k, v in vers.items() if v["version"] == max((x["version"] for x in vers.values()), key=lambda s: int(s.split(".")[1]))}
srcs = set(REUSE_DEFAULT_SOURCES) | {l for v in REUSE_FROM.values() for l in v}
seen, out = set(), []
for d in dirs:
    for f in glob.glob(str(HERE / d / "*.raw.jsonl")):
        lang = os.path.basename(f).split(".")[0]
        if lang not in srcs: continue
        for l in open(f, encoding="utf-8").read().split("\n"):
            if not l.strip(): continue
            r = json.loads(l)
            if not (r.get("eng") and r.get("m_text") and r.get("hm") and r.get("ood")): continue
            if not allv and r.get("prompt_fp") not in cur: continue
            if r.get("source_row"): continue                     # never chain D rows
            g = gates.check(dict(r), pl.P[lang])
            if g["fail"]: continue
            h = hashlib.md5(r["eng"].encode()).hexdigest()
            if h in seen: continue
            seen.add(h); out.append({"id": r["id"], "lang": lang, "eng": r["eng"], "genre": r["scenario"]["genre"], "domain": r["scenario"]["domain"], "kind": "statement"})
            # one-sentence records for the single-sentence languages: plain, self-contained sentences of 5-15 words
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", r["eng"]) if 5 <= len(s.split()) <= 15]
            sents = [s for s in sents if not re.match(r"(he|she|it|they|this|that|these|those|and|but|so|then|also|however)\b", s, re.I) and '"' not in s]
            for k, s in enumerate(sents[:2]): out.append({"id": f"{r['id']}#s{k}", "lang": lang, "eng": s, "genre": "chat", "domain": r["scenario"]["domain"], "kind": "sentence"})
(HERE / "eng_pool.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + ("\n" if out else ""), encoding="utf-8")
print(len(out), "English statements in the pool from", len({o["lang"] for o in out}), "languages ->", "eng_pool.jsonl")
