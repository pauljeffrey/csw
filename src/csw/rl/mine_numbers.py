"""Number words per language, mined from the SFT eng_m (English -> language) pairs by association: for each English number word (one..ten, twenty..hundred, thousand), the target-language words that occur in far more of the sentences containing it than in the language's other sentences (ratio >= 12, support >= 4, document-frequency cap).  Written to data/number_words.json: {lang: [words]}.
Training data used: final_v5/train.jsonl eng_m rows only (the SFT data, read-only; nothing is trained here).   python -m csw.rl.mine_numbers --sft final_v5/train.jsonl --out csw_rl/data/number_words.json"""
import argparse, collections, json, re, unicodedata

NUMS = "one two three four five six seven eight nine ten twenty thirty forty fifty hundred thousand".split()
W = re.compile(r"\w+", re.U)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sft", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ratio", type=float, default=12.0); ap.add_argument("--support", type=int, default=4); a = ap.parse_args()
    rows = collections.defaultdict(list)
    for l in open(a.sft, encoding="utf-8"):
        if '"eng_m"' not in l[:300]: continue
        r = json.loads(l)
        if r.get("task") != "eng_m": continue
        src = r["prompt"].split(":", 1)[-1]; tgt = re.sub(r"^[a-z]{3}\s+", "", r["completion"]); rows[r["lang"]].append((set(W.findall(src.lower())), set(W.findall(unicodedata.normalize("NFC", tgt.lower())))))
    out = {}
    for lang, rs in rows.items():
        n = len(rs); df = collections.Counter(w for _, t in rs for w in t); res = set()
        for num in NUMS:
            sel = [t for s, t in rs if num in s]
            if len(sel) < a.support: continue
            c = collections.Counter(w for t in sel for w in t)
            for w, k in c.items():
                if k >= a.support and not w.isdigit() and len(w) >= 2 and (k / len(sel)) / (df[w] / n) >= a.ratio and k / len(sel) >= 0.35: res.add(w)
        out[lang] = sorted(res); print(lang, n, len(res), sorted(res)[:12])
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
