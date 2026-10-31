"""Reference-free AfriCOMET-1.1 (masakhane/africomet-qe-stl-1.1) scores for (english, translation) pairs.  Run with the COMET venv.
python africomet_score.py --pairs r4/nb_pool.jsonl --out r4/nb_scored.jsonl [--bs 64]"""
import argparse, json, time
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--half", action="store_true", help="fp16 weights (about half the GPU memory)"); a = ap.parse_args()
from comet import download_model, load_from_checkpoint
rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
if a.limit: rows = rows[: a.limit]
model = load_from_checkpoint(download_model("masakhane/africomet-qe-stl-1.1"))
if a.half: model = model.half()
t0 = time.time()
out = model.predict([{"src": r["eng"], "mt": r["m"]} for r in rows], batch_size=a.bs, gpus=1, progress_bar=False)
with open(a.out, "w", encoding="utf-8") as f:
    for r, s in zip(rows, out.scores): f.write(json.dumps(dict(r, qe=float(s)), ensure_ascii=False) + "\n")
print(f"scored {len(rows)} pairs in {time.time()-t0:.0f}s ({len(rows)/(time.time()-t0):.0f}/s)")
