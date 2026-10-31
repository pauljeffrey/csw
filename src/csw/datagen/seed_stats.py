"""Seed statistics per language -> seed_stats.md  (python -m csw.datagen.seed_stats)"""
from csw.paths import DATA_ROOT
import json, re, glob, os, statistics as st, collections as C, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen.profiles import P, SUPPORT
W = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.U)
def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q / 100; f = int(k); c = min(f + 1, len(v) - 1); return v[f] + (v[c] - v[f]) * (k - f)
MT = {"ultrachat", "translated", "mt560", "sentence", "pairs", "nllb", "opus", "alpaca", "instruct", "sft", "smol", "flores", "ntrex", "xp3", "massivesumm", "tatoeba", "parallel", "madlad"}
REL = {"jw", "bible", "verse", "verses", "gospel", "hymn", "sermon", "quran", "watchtower", "jehovah"}
rows, tot_n = [], C.Counter(); shares = {}
for lang in P:
    docs = []
    for name in (f"{lang}.vast.jsonl", f"{lang}.jsonl"):
        p = HERE / "seeds" / name
        if p.exists():
            docs += [json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()]
    docs = [d for d in docs if d.get("track") == "B"]
    anch = sum(1 for _ in open(HERE / "anchors" / f"{lang}.jsonl", encoding="utf-8")) if (HERE / "anchors" / f"{lang}.jsonl").exists() else 0
    if not docs:
        rows.append((lang, 0, anch)); continue
    wc = [len(W.findall(d["text"])) for d in docs]
    ws = [w for d in docs for w in W.findall(d["text"])]
    cl = sum(len(w) for w in ws) / max(1, len(ws))
    toks = [set(re.split(r"[^a-z0-9]+", (d.get("src") or "").lower())) for d in docs]
    mt = sum(1 for t in toks if t & MT); rel = sum(1 for t in toks if t & REL or (d := None) is None and False)
    rel = sum(1 for d, t in zip(docs, toks) if (t & REL) or d.get("bucket") == "religion")
    mt = sum(1 for d, t in zip(docs, toks) if (t & MT) or d.get("bucket") in ("qa", "pairs"))
    b = C.Counter(d.get("bucket", "hub") for d in docs)
    rows.append((lang, len(docs), anch, st.mean(wc), pct(wc, 50), min(wc), pct(wc, 5), pct(wc, 25), pct(wc, 75), pct(wc, 95), max(wc), cl, rel / len(docs), mt / len(docs), b))
    shares[lang] = b
tier = lambda l: next((t for t, v in SUPPORT.items() if l in v), "-")
out = ["# Seed statistics (native-language seeds, track B)\n", "Words = alphabetic word tokens per seed document. 'avg chars/word' = mean letters per word. religion% / MT-ish% = share of seeds from religious sources or from translated / instruction / parallel-pair sources (by source name and bucket).\n",
       "| lang | tier | seeds | anchors | words mean | median | min | p5 | p25 | p75 | p95 | max | chars/word | religion% | MT-ish% |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r[1] == 0: out.append(f"| {r[0]} | {tier(r[0])} | 0 | {r[2]} | - | - | - | - | - | - | - | - | - | - | - |"); continue
    l, n, a, mean, med, mn, p5, p25, p75, p95, mx, cl, rel, mt, b = r
    out.append(f"| {l} | {tier(l)} | {n} | {a} | {mean:.0f} | {med:.0f} | {mn} | {p5:.0f} | {p25:.0f} | {p75:.0f} | {p95:.0f} | {mx} | {cl:.2f} | {rel:.0%} | {mt:.0%} |")
allw = []; tot = sum(r[1] for r in rows)
for lang in P:
    pass
ar = [r for r in rows if r[1]]
out.append(f"\n**Totals:** {tot:,} seeds over {len(ar)} languages; per-language seeds: mean {st.mean(r[1] for r in ar):.0f}, median {st.median(r[1] for r in ar):.0f}, min {min(r[1] for r in ar)} ({min(ar,key=lambda r:r[1])[0]}), max {max(r[1] for r in ar)} ({max(ar,key=lambda r:r[1])[0]}); anchors total {sum(r[2] for r in rows):,}.")
bt = C.Counter()
for b in shares.values(): bt.update(b)
out.append("\n**Seeds by source type (all languages):** " + ", ".join(f"{k} {v:,} ({v/tot:.0%})" for k, v in bt.most_common()))
out.append(f"\n**Weighted share:** religion {sum(r[1]*r[12] for r in ar)/tot:.0%}, MT-ish/instruction/pairs {sum(r[1]*r[13] for r in ar)/tot:.0%}.")
Path(HERE / "seed_stats.md").write_text("\n".join(out), encoding="utf-8"); print("\n".join(out[-5:])); 
# compact console table
print("\nlang seeds anchors | words mean/median/min/p5/p95/max | chars/word | rel% mt%")
for r in rows:
    if r[1]: print(f"{r[0]:4}{r[1]:6}{r[2]:5} | {r[3]:5.0f} {r[4]:5.0f} {r[5]:4} {r[6]:5.0f} {r[9]:5.0f} {r[10]:6} | {r[11]:.2f} | {r[12]:.0%} {r[13]:.0%}")
    else: print(f"{r[0]:4}{0:6}{r[2]:5} | (no native seeds)")
