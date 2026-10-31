"""Merge the hm repair into the rows WITHOUT discarding the old values.
python -m csw.sft.merge_repairs --raw DIR_WITH_raw_and_repair --out OUT
For every language writes OUT/<lang>.raw.jsonl = all original rows (unchanged, old hm kept) + one NEW row per repair that passed the gate:
id <orig>-hm2, hm = repaired hm, hm_rejected = old hm, parent_id = orig id, quality_tags ["hm_v2"]; the original gets quality_tags ["hm_english_leak_v1"] when it had a repair record."""
import argparse, glob, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tot = {"orig": 0, "variants": 0, "tagged": 0, "failed": 0}
for f in sorted(glob.glob(f"{a.raw}/*.raw.jsonl")):
    lang = Path(f).name.split(".")[0]; rep = {}
    rp = Path(a.raw) / f"{lang}.repair.jsonl"
    if rp.exists():
        for l in open(rp, encoding="utf-8"):
            if l.strip():
                x = json.loads(l); rep[x["id"]] = x
    rows = []
    for l in open(f, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l); tot["orig"] += 1; x = rep.get(r["id"])
        if x:
            r = dict(r, quality_tags=list(r.get("quality_tags") or []) + ["hm_english_leak_v1"]); tot["tagged"] += 1
        rows.append(r)
        if x and x.get("passed") and x.get("hm_new"):
            v = dict(r, id=r["id"] + "-hm2", parent_id=r["id"], hm_rejected=r["hm"], hm=x["hm_new"], quality_tags=["hm_v2"]); rows.append(v); tot["variants"] += 1
        elif x:
            tot["failed"] += 1
    (out / f"{lang}.raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
print(tot)
