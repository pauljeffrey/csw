"""Which languages does the generator actually handle?  python -m csw.datagen.lang_support out_gb10
For each language: build a word list from the REAL seed text (held out 10% of seed docs as a control), then measure the
share of words in generated m_text / ood that never occur in real text of that language (OOV), relative to the same measure
on held-out REAL text. ratio ~1 = as natural as real text; >>1 = invented/garbled words. Agglutinative languages have high OOV
even for real text, which is why the control is used."""
from csw.paths import DATA_ROOT
import json, re, sys, glob, os, random, statistics as st
from pathlib import Path
HERE = DATA_ROOT
W = re.compile(r"[^\W\d_]+", re.U); tk = lambda t: [w.lower() for w in W.findall(t)]
od = Path(sys.argv[1]); out = {}
for f in sorted(glob.glob(str(od / "*.raw.jsonl"))):
    lang = os.path.basename(f).split(".")[0]
    seeds = []
    for name in (f"{lang}.vast.jsonl", f"{lang}.jsonl"):
        p = HERE / "seeds" / name
        if p.exists():
            seeds += [json.loads(x)["text"] for x in p.read_text(encoding="utf-8").split("\n") if x.strip()]
    if len(seeds) < 60:
        out[lang] = None; continue
    random.Random(1).shuffle(seeds); k = max(10, len(seeds) // 10)
    held, train = seeds[:k], seeds[k:]
    vocab = set(w for t in train for w in tk(t))
    def oov(texts):
        ws = [w for t in texts for w in tk(t)]
        ws = [w for w in ws if len(w) > 2]
        return sum(1 for w in ws if w not in vocab) / max(1, len(ws)), len(ws)
    real, _ = oov(held)
    rows = [json.loads(x) for x in open(f, encoding="utf-8").read().split("\n") if x.strip()]
    m, mn = oov([r["m_text"] for r in rows if r.get("m_text")]); o, on = oov([r["ood"] for r in rows if r.get("ood")])
    out[lang] = {"rows": len(rows), "seed_docs": len(seeds), "oov_real": round(real, 3), "oov_m_text": round(m, 3), "oov_ood": round(o, 3),
                 "ratio_m": round(m / max(real, 1e-3), 2), "ratio_ood": round(o / max(real, 1e-3), 2)}
json.dump(out, open(HERE / "lang_support.json", "w"), indent=1)
print(f"{'lang':5}{'rows':>5}{'real':>7}{'m_text':>8}{'ood':>7}{'ratio_m':>9}{'ratio_ood':>10}")
for l, v in sorted(out.items(), key=lambda kv: (kv[1] is None, kv[1]["ratio_m"] if kv[1] else 0)):
    print(f"{l:5}" + (f"{v['rows']:>5}{v['oov_real']:>7.2f}{v['oov_m_text']:>8.2f}{v['oov_ood']:>7.2f}{v['ratio_m']:>9.2f}{v['ratio_ood']:>10.2f}" if v else "   (too few seeds)"))
