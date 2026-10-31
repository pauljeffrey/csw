"""Simple English seeds: TinyStories ranked by Flesch-Kincaid grade, lowest (simplest) kept.   python -m csw.datagen.tiny_fk [n_fetch=2000] [keep=400]"""
from csw.paths import DATA_ROOT
import json, re, sys, hashlib, statistics as st
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import seeds
n, keep = int(sys.argv[1]) if len(sys.argv) > 1 else 2000, int(sys.argv[2]) if len(sys.argv) > 2 else 400
def syll(w):
    w = w.lower(); g = re.findall(r"[aeiouy]+", w); c = len(g)
    if w.endswith("e") and not w.endswith(("le", "ee")) and c > 1: c -= 1
    return max(1, c)
def fk(t):
    s = max(1, len(re.findall(r"[.!?]+", t))); ws = re.findall(r"[A-Za-z']+", t); w = max(1, len(ws)); sy = sum(syll(x) for x in ws)
    return 0.39 * w / s + 11.8 * sy / w - 15.59
rows = seeds.fetch_rows("roneneldan/TinyStories", n, seed=11)
docs = []
for r in rows:
    t = re.sub(r"\s+", " ", r["text"]).strip(); wc = len(t.split())
    if 30 <= wc <= 130: docs.append((fk(t), wc, t))
docs.sort(); g = [d[0] for d in docs]
print(f"fetched {len(rows)}, usable {len(docs)}; Flesch-Kincaid grade: min {g[0]:.1f}  p10 {g[len(g)//10]:.1f}  median {st.median(g):.1f}  p90 {g[9*len(g)//10]:.1f}  max {g[-1]:.1f}")
sel = docs[:keep]
print(f"kept the {len(sel)} simplest: grade {sel[0][0]:.1f} .. {sel[-1][0]:.1f}; words mean {st.mean(d[1] for d in sel):.0f}")
with open(HERE / "seeds" / "_en_simple.jsonl", "w", encoding="utf-8") as f:
    for gr, wc, t in sel:
        f.write(json.dumps({"seed_id": hashlib.md5(t.encode()).hexdigest()[:12], "lang": "_en", "track": "C", "genre": "story", "text": t, "src": "roneneldan/TinyStories", "license": "cdla-sharing-1.0", "fk_grade": round(gr, 2), "words": wc}, ensure_ascii=False) + "\n")
for label, d in (("simplest", sel[0]), ("kept median", sel[len(sel)//2]), ("rejected (median story)", docs[len(docs)//2])):
    print(f"\n[{label}] FK {d[0]:.1f}, {d[1]} words:\n  {d[2][:330]}")
