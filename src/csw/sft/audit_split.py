"""Leak audit of an SFT build: test (and val) rows must share nothing with train.
    python -m csw.sft.audit_split --sft work/csw_sft_v3 [--sets test val]
Checks per held-out example against ALL train examples: (1) same source row (id root, judge copies -fxf/-fxl and repairs -hm2 count as the same row),
(2) identical prompt text, (3) identical final completion text (tags and devices removed), (4) any shared word 8-gram in the source text or in the final text."""
import argparse, collections, json, re
from pathlib import Path


def root(i): return re.sub(r"(-fx[fl]|-hm2)+$", "", i.split(":")[0])
def final(c): return " ".join(re.sub(r"<[^>]{1,12}>", " ", re.sub(r"<dev>.*?</dev>", " ", c, flags=re.S)).split())
def norm(t): return re.sub(r"[\W_]+", " ", t.lower()).strip()
def src(p): return p.split(":", 1)[1].strip() if ":" in p else p
def grams(t, n=8):
    w = norm(t).split(); return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sft", required=True); ap.add_argument("--sets", nargs="+", default=["test", "val"]); a = ap.parse_args(); d = Path(a.sft)
    roots, prompts, finals, g_src, g_fin = set(), set(), set(), set(), set(); n = 0
    for l in open(d / "train.jsonl", encoding="utf-8"):
        e = json.loads(l); n += 1; roots.add(root(e["id"])); prompts.add(norm(e["prompt"])); f = final(e["completion"]); finals.add(norm(f))
        if e["task"] != "eng_m": g_src |= grams(src(e["prompt"])); g_fin |= grams(f)
        else: g_src |= grams(src(e["prompt"]))
    print("train examples", n, "roots", len(roots))
    for s in a.sets:
        c = collections.Counter(); m = 0; bad = collections.Counter()
        for l in open(d / f"{s}.jsonl", encoding="utf-8"):
            e = json.loads(l); m += 1; f = final(e["completion"]); hit = []
            if root(e["id"]) in roots: hit.append("row")
            if norm(e["prompt"]) in prompts: hit.append("prompt")
            if norm(f) in finals: hit.append("completion")
            if e["task"] != "eng_m" and (grams(src(e["prompt"])) & g_src): hit.append("src_8gram")
            if e["task"] != "eng_m" and (grams(f) & g_fin): hit.append("final_8gram")
            for h in hit: c[h] += 1
            if hit: bad[e["lang"]] += 1
        print(s, m, "examples; leaks:", dict(c) or "none", "| languages with a leak:", dict(bad.most_common(8)))


if __name__ == "__main__":
    main()
