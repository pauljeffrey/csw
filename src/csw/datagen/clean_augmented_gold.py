"""Tolerant loader + cleaner for the hand-edited exemplars/augmented_gold.jsonl.
Splits on object starts, repairs the usual hand-edit slips (stray metadata after the object, a missing comma before a trailing key, `{""lang"`),
removes `status` and the machine provenance keys, writes one valid JSON object per line. The untouched original is kept as *.original.jsonl."""
from csw.paths import DATA_ROOT
import json, re, sys, collections as C
from pathlib import Path
HERE = DATA_ROOT; src = HERE / "exemplars" / "augmented_gold.original.jsonl"; dst = HERE / "exemplars" / "augmented_gold.jsonl"
DROP = {"status", "auto_flags", "generator_support", "prompt_version"}
lines = src.read_text(encoding="utf-8").split("\n")
starts = [i for i, l in enumerate(lines) if re.match(r'^\s*\{\s*"{0,2}lang"', l)]
blocks = ["\n".join(lines[a:b]) for a, b in zip(starts, starts[1:] + [len(lines)])]
def parse(b):
    b = re.sub(r'^\s*\{\s*"{0,2}lang"', '{"lang"', b.strip(), count=1)
    if b.count("[") > b.count("]"):          # a list was closed with a comma instead of ]
        b = re.sub(r'(\n\s*),\s*"(generator_support|source_id|dialect|prompt_version|auto_flags)"', r'\1], "\2"', b, count=1)
    for attempt in range(4):
        try:
            o, end = json.JSONDecoder().raw_decode(b); return o, "ok" if not b[end:].strip() else "trailing-ignored"
        except json.JSONDecodeError as e:
            if "delimiter" in e.msg and e.msg.startswith("Expecting ','"):
                pos = e.pos; b = b[:pos].rstrip() + "," + b[pos:] if pos else b      # insert the missing comma where the parser stopped
                b = re.sub(r'\]\s*\n\s*"', '],\n"', b, count=1) if attempt else b
            else:
                fixed = re.sub(r'(\n\s*),\s*"(generator_support|source_id|dialect)"', r'\1], "\2"', b, count=1)   # list closed with a comma instead of ]
                if fixed == b: break
                b = fixed
    return None, "unrecoverable"
rows, log = [], C.Counter()
for k, b in enumerate(blocks):
    o, how = parse(b); log[how] += 1
    if o is None or not all(o.get(f) for f in ("lang", "eng", "m_text", "hm", "ood")): print("SKIPPED block", k, "line", starts[k] + 1, how, (o or {}).get("lang")); continue
    rows.append({k2: v for k2, v in o.items() if k2 not in DROP})
print(dict(log), "->", len(rows), "clean rows")
retired = [r for r in rows if r["lang"] == "urh"]; rows = [r for r in rows if r["lang"] != "urh"]
(HERE / "retired").mkdir(exist_ok=True)
(HERE / "retired" / "augmented_gold_urh.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in retired) + "\n", encoding="utf-8")
dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
n = C.Counter(r["lang"] for r in rows)
print(f"{len(rows)} rows, {len(n)} languages written to {dst.name}; Urhobo rows ({len(retired)}) moved to retired/")
print("rows per language:", dict(sorted(n.items(), key=lambda x: -x[1])))
print("keys kept:", dict(C.Counter(k for r in rows for k in r).most_common()))
