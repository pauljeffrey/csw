"""Turn judge_pool.jsonl + rows into judge requests (natural rows, no ladders) for judge_pilot2 run.
python -m csw.judging.build_pool_requests --pool P --rows DIR --out OUTDIR --splits eval labeled [--skip-langs a b]"""
import argparse, glob, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--splits", nargs="+", default=["eval", "labeled"]); ap.add_argument("--skip-langs", nargs="*", default=[]); a = ap.parse_args()
pool = [json.loads(l) for l in open(a.pool, encoding="utf-8") if l.strip()]
want = {p["id"]: p for p in pool if p["split"] in a.splits and p["lang"] not in a.skip_langs}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n = 0
with open(out / "requests.jsonl", "w", encoding="utf-8") as fo:
    for f in sorted(glob.glob(f"{a.rows}/*.raw.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r["id"] in want:
                    p = want[r["id"]]; fo.write(json.dumps({"rid": r["id"], "kind": "natural", "lang": r["lang"], "tier": p["tier"], "split": p["split"], "pair": p.get("pair"), "row": r}, ensure_ascii=False) + "\n"); n += 1
print(n, "requests ->", out)
