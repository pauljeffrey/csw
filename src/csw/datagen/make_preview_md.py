"""python -m csw.datagen.make_preview_md <out_dir> <out.md>  - 5 rows per language, Nigerian languages first, with gate flags and the generator-support tier."""
import json, sys, glob, os, collections as C
from csw.datagen import gates
from csw.datagen import cs_pipeline as pl
from csw.datagen.profiles import SUPPORT, P
od, outp = sys.argv[1], sys.argv[2]
NIG = ["yor", "hau", "ibo", "pcm", "efi", "ibb", "urh", "bin"]
have = {os.path.basename(f).split(".")[0]: f for f in glob.glob(f"{od}/*.raw.jsonl")}
order = [l for l in NIG if l in have] + [l for l in P if l in have and l not in NIG]
tier = lambda l: next((t for t, v in SUPPORT.items() if l in v), "unmeasured")
fp = set(); cnt = C.Counter(); n = 0
md = ["# Code-switching preview - 5 rows per language, current prompt\n",
      "Unedited rows from Gemma-4-31B (vLLM fp8) on the GB10 box. `eng` = formal official English; `m_text` = its official translation; `hm` = informal code-switched variation; `ood` = informal native parlance (may code-switch).",
      "`gates` = deterministic checks (fail = would be dropped, warn = flagged). `generator support` = how well Gemma handles the language (ok / borderline / dropped), measured by out-of-vocabulary rate vs real text (see DESIGN.md / lang_support.py). Nobody has judged correctness yet - a native reader or the LLM judge must.\n"]
md.append("## Index\n"); md.append("| language | generator support | rows |\n|---|---|---|")
for l in order: md.append(f"| [{l}](#{l}) {P[l].name} | {tier(l)} | {sum(1 for _ in open(have[l]))} |")
for l in order:
    rows = [json.loads(x) for x in open(have[l], encoding="utf-8").read().split("\n") if x.strip()]
    md.append(f"\n## {l}\n**{P[l].name}** - generator support: **{tier(l)}** - switch languages: {', '.join(P[l].switch)}\n")
    for r in rows:
        s = r["scenario"]; g = gates.check(dict(r), P[l]); fp.add(r.get("prompt_fp")); n += 1
        for w in g["fail"] + g["warn"]: cnt[w] += 1
        md.append(f"### {s['genre']} / {s['length']} / {s['domain']} - {s['dialect']} + **{s['switch']}** - {s['persona']} to {s['audience']} - track {s['track']}\ngates: fail={g['fail']} warn={g['warn']}\n")
        for k in ("eng", "m_text", "hm", "ood"): md.append(f"- **{k}**: {r.get(k)}")
        md.append(f"- devices: {r.get('devices')}\n")
md.insert(1, f"Prompt fingerprint(s): {sorted(str(x) for x in fp)} - {n} rows - gate flags overall: {dict(cnt.most_common())}\n")
open(outp, "w", encoding="utf-8").write("\n".join(md)); print(n, "rows ->", outp, dict(cnt.most_common(8)))
