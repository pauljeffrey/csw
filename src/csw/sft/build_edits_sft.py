"""csw SFT examples from the strict-Flash corrected rows: apply the judge's edits to hm / ood (all-or-nothing per field, with csw.sft.apply_edits' post-checks), rebuild the five tasks with the SFT builder.
python -m csw.sft.build_edits_sft --records work/stage2_both/records.jsonl --rows work/post_rows --out work/csw_edits"""
from csw.paths import DATA_ROOT
import argparse, collections, copy, json, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.sft import build_sft as B
from csw.sft import devices as dv
from csw.sft.apply_edits import apply_edits
from csw.sft import build_sft2 as S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", required=True); ap.add_argument("--rows", default="work/post_rows"); ap.add_argument("--out", required=True); ap.add_argument("--lexicon", default="work/lexicon.json"); a = ap.parse_args()
    rec = {json.loads(l)["id"]: json.loads(l) for l in open(a.records, encoding="utf-8")}; rows = {r["id"]: r for r in B.load_rows(a.rows, []) if r["id"] in rec}
    lex = dv.Lexicon.load(a.lexicon); st = collections.Counter(); out = []; EVAL_SHAS = set()
    for rid, x in rec.items():
        r = rows.get(rid)
        if not r: st["row_missing"] += 1; continue
        r = copy.deepcopy(r); changed = False
        for field, key in (("h", "hm"), ("o", "ood")):
            ed = [{"find": e[1], "ctx": "", "rep": e[2]} for e in (x.get("edits") or []) if isinstance(e, list) and len(e) == 3 and e[0] == field and isinstance(e[1], str) and isinstance(e[2], str)]
            if not ed: continue
            new, rep = apply_edits(r[key], ed)
            if rep.get("ok") and new != r[key]: r[key] = new; changed = True; st[f"applied_{key}"] += 1
            else: st[f"failed_{key}"] += 1; st["why:" + str(rep.get("why", "unchanged"))[:40]] += 1
        if not changed: st["no_applied_edit"] += 1; continue
        r["id"] = rid + "-fxe"; ex = S.examples(r, lex, st)
        for e in ex:
            if e["split"] == "train": out.append(e)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "train.jsonl", "w", encoding="utf-8") as f:
        for e in out: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(dict(st), "| examples", len(out), dict(collections.Counter(e["task"] for e in out)))


if __name__ == "__main__":
    main()
