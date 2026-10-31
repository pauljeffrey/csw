"""Keep every generated row and TAG it:  python -m csw.datagen.tag_rows <out_dir> [lang ...]  ->  <lang>.tagged.jsonl
Adds: prompt_version (from prompt_fp via prompt_versions.json; rows with no fingerprint are v3.0), superseded (not made by the
current prompt), quality_tags (deterministic gate failures/warnings + hm over-mixing/tag-only), gate_metrics.
Nothing is deleted or rewritten in the raw files; downstream steps filter on the tags (or regenerate the tagged field)."""
from csw.paths import DATA_ROOT
import difflib, json, re, sys, glob, os, collections as C
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import gates
from csw.datagen import cs_pipeline as pl
od = Path(sys.argv[1]); langs = sys.argv[2:] or sorted({os.path.basename(f).split(".")[0] for f in glob.glob(f"{od}/*.raw.jsonl")})
vers = json.loads((HERE / "prompt_versions.json").read_text())
current = max(vers.values(), key=lambda v: int(v["version"].split(".")[1]))["version"]
W = re.compile(r"[^\W\d_]+", re.U); tk = lambda t: [w.lower() for w in W.findall(t)]
tot = C.Counter(); bylang = {}
for lang in langs:
    rows = [json.loads(x) for x in (od / f"{lang}.raw.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    out, cnt = [], C.Counter()
    for r in rows:
        v = vers.get(r.get("prompt_fp") or "none", {"version": "v3.?"})["version"]
        g = gates.check(dict(r), pl.P[lang])
        tags = [f"fail:{x}" for x in g["fail"]] + [f"warn:{x}" for x in g["warn"]]
        e = set(tk(r["eng"])); h = tk(r["hm"])
        share = sum(1 for w in h if w in e and w.isascii() and len(w) > 2) / max(1, len(h))
        if lang != "pcm" and share > 0.30: tags.append("hm_over_mixed")
        if lang == "pcm" and difflib.SequenceMatcher(None, tk(r["m_text"]), h).ratio() > 0.85: tags.append("hm_tags_only")
        r2 = dict(r, prompt_version=v, superseded=(v != current), quality_tags=tags, gate_metrics=g["metrics"])
        out.append(r2); cnt[v] += 1
        for t in tags: tot[t] += 1
    (od / f"{lang}.tagged.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    bylang[lang] = dict(cnt)
print("current prompt version:", current); print("rows by version per language:", bylang if len(bylang) < 8 else {k: sum(v.values()) for k, v in bylang.items()})
print("tag counts:", dict(tot.most_common(14)))
