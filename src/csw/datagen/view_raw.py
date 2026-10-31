"""Readable markdown from raw generated rows (no judge needed).  python -m csw.datagen.view_raw out_gb10 samples_gb10.md [rows_per_lang]"""
import json, sys, glob, os, collections as C
from csw.datagen import gates
from csw.datagen import cs_pipeline as pl
od, outp, k = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4
md = ["# African-Code-Switch - raw rows from the GB10 run (Gemma-4-31B, vLLM fp8, prompt v3 + 6 gold rows)\n",
      "Unjudged, uncorrected rows exactly as generated. `gates` = deterministic checks (fail = would be dropped, warn = flagged).\n"]
tot = C.Counter(); n = 0
for f in sorted(glob.glob(f"{od}/*.raw.jsonl")):
    rows = [json.loads(x) for x in open(f, encoding="utf-8").read().split("\n") if x.strip()]
    lang = os.path.basename(f).split(".")[0]
    md.append(f"\n## {lang} ({len(rows)} rows so far)\n")
    for r in rows:
        g = gates.check(dict(r), pl.P[lang]); n += 1
        for w in g["fail"] + g["warn"]: tot[w] += 1
    for r in sorted(rows, key=lambda r: (r["scenario"]["length"], len(r["eng"])))[:: max(1, len(rows) // k)][:k]:
        s = r["scenario"]; g = gates.check(dict(r), pl.P[lang])
        md.append(f"**{s['genre']} / {s['length']} / {s['domain']}** - track {s['track']} - {s['dialect']} - switch **{s['switch']}** - speaker: {s['persona']} to {s['audience']} - gates: fail={g['fail']} warn={g['warn']}\n")
        for key in ("eng", "m_text", "hm", "ood"): md.append(f"- **{key}**: {r.get(key)}")
        md.append(f"- devices: {r.get('devices')}  conf: {r.get('conf')}  tokens(in/out): {r.get('usage')}\n")
open(outp, "w", encoding="utf-8").write("\n".join(md)); print("rows", n, dict(tot), "->", outp)
