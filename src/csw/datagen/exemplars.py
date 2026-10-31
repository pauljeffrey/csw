"""Build exemplar banks from adjudicated rows: python -m csw.datagen.exemplars <lang> <out_dir> [k] [min_score].
Picks the highest final_score kept rows (corrected fields), short/medium only, different genres."""
from csw.paths import DATA_ROOT
import json, sys
from pathlib import Path
HERE = DATA_ROOT
lang, od = sys.argv[1], HERE / sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 6
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
rows = [json.loads(x) for x in (od / f"{lang}.final.g.jsonl").read_text().splitlines() if x.strip()]
rows = [r for r in rows if r.get("judge_scores") and r["final_score"] >= thr and r["scenario"]["length"] != "long" and r.get("gates_final", {}).get("fail") == []]
rows.sort(key=lambda r: -r["final_score"])
out, genres = [], set()
for r in rows:
    g = r["scenario"]["genre"]
    if g in genres and len(rows) > k:
        continue
    genres.add(g)
    f = r["final"]
    out.append({"eng": r["eng"], "m_text": f["m_text"], "hm": f["hm"], "ood": f["ood"], "ood_relation": r.get("ood_relation"), "devices": r.get("devices"),
                "switch": r["scenario"]["switch"], "genre": g, "score": r["final_score"], "src": r.get("teacher")})
    if len(out) >= k:
        break
(HERE / "exemplars").mkdir(exist_ok=True)
(HERE / "exemplars" / f"{lang}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out))
print(lang, len(out), [round(x["score"], 2) for x in out])
