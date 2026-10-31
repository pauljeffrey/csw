"""Render judged rows into a readable markdown file + print structure statistics.  python -m csw.datagen.view_samples out_v3 samples_v3.md"""
import json, sys, statistics as st, collections as C
from pathlib import Path
od, outp = Path(sys.argv[1]), sys.argv[2]
rows = []
for f in sorted(od.glob("*.judged.g.jsonl")):
    rows += [json.loads(x) for x in f.read_text(encoding="utf-8").split("\n") if x.strip()]
tot = len(rows); full = [r for r in rows if all(r.get(k) for k in ("eng", "m_text", "hm", "ood"))]
print(f"rows {tot}; all 4 fields present {len(full)}; gate-fail {sum(1 for r in rows if r['gates']['fail'])}")
fails = C.Counter(x for r in rows for x in r['gates']['fail']); warns = C.Counter(x for r in rows for x in r['gates']['warn'])
print("gate fails", dict(fails)); print("gate warns", dict(warns))
dims = C.defaultdict(list)
for r in rows:
    for k, v in ((r.get("judge") or {}).get("scores") or {}).items():
        if isinstance(v, (int, float)): dims[k].append(v)
print("judge means:", {k: round(st.mean(v), 2) for k, v in dims.items()})
sw = [r['gates']['metrics'].get('sw_hm') for r in full if r['gates'].get('metrics')]; swo = [r['gates']['metrics'].get('sw_ood') for r in full if r['gates'].get('metrics')]
print("hm switch-word share mean %.2f  | ood %.2f" % (st.mean([x for x in sw if x is not None]), st.mean([x for x in swo if x is not None])))
byl = C.defaultdict(list)
for r in rows:
    s = [v for v in ((r.get("judge") or {}).get("scores") or {}).values() if isinstance(v, (int, float))]
    if s: byl[r['lang']].append(sum(s) / len(s))
print("per-language judge mean:", {k: round(st.mean(v), 2) for k, v in sorted(byl.items(), key=lambda x: -st.mean(x[1]))})
md = ["# Code-switch corpus - v3 sample (Gemma-4-31B, 3 rows per language)\n", "Fields: `eng` plain official English - `m_text` official translation - `hm` natural code-switched variation - `ood` full informal native parlance. Judge = DeepSeek-V4-Pro (not a native speaker). Rows shown BEFORE judge corrections unless marked.\n"]
for lang in sorted({r['lang'] for r in rows}):
    md.append(f"\n## {lang}\n")
    for r in [x for x in rows if x['lang'] == lang]:
        s = r['scenario']; j = (r.get('judge') or {}).get('scores') or {}
        md.append(f"**{s['genre']} / {s['length']} / {s['domain']}** - track {s['track']} - dialect {s['dialect']} - switch **{s['switch']}** - to {s.get('audience','')} - gates: fail={r['gates']['fail']} warn={r['gates']['warn']} - judge: " + ", ".join(f"{k}={v}" for k, v in j.items()) + "\n")
        for k in ("eng", "m_text", "hm", "ood"):
            md.append(f"- **{k}**: {r.get(k)}")
        md.append(f"- devices: {r.get('devices')}  conf: {r.get('conf')}\n")
Path(outp).write_text("\n".join(md), encoding="utf-8"); print("wrote", outp, len("\n".join(md)))
