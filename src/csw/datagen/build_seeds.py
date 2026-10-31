"""Run on the GPU/CPU box: sample slices -> decode -> split documents -> clean/filter -> seed JSONL per language.

  python -m csw.datagen.build_seeds OUTDIR TARGET LANG [LANG ...]

Seeds are INSPIRATION for a new situation, never copied text, so the filters aim at: really that language, a few
sentences long, no URLs/contacts, not boilerplate, not near-duplicate, spread over source buckets (genre).
"""
import hashlib, json, os, random, re, sys
import numpy as np
from csw.datagen import extract_slices as ex
from tokenizers import Tokenizer

TOK = Tokenizer.from_file(os.environ.get("CSW_TOKENIZER_JSON", "tokenizer.json"))
EN = set("the a an and or but of to in on at for with from by is are was were be been being have has had do does did not this that these those it its i you he she we they me him her us them my your his our their what which who when where why how if then than so very just also can could will would shall should may might must about into over after before again more most some any all each every too only own same such there here".split())
FR = set("le la les un une des et ou mais de du au aux en dans sur avec pour par est sont était être avoir ai as a avons avez ont ne pas non oui ce cet cette ces il elle nous vous ils elles je tu me te se mon ma mes ton ta tes son sa ses notre votre leur qui que quoi dont où quand comment pourquoi si alors très aussi plus moins bien donc".split())
GENRE = {"chat": "chat", "social": "chat", "story": "story", "news": "opinion", "qa": "explain_teach", "proverb": "tradition_law",
         "religion": "faith", "pairs": "chat", "general": "explain_teach"}
URL = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+|\+?\d[\d\s\-()]{8,}\d|@\w+)")
WORD = re.compile(r"[^\W\d_]+", re.U)
BOILER = re.compile(r"(cookie|copyright|all rights reserved|privacy policy|subscribe|click here|read more|login|sign up|terms of (use|service)|lorem ipsum)", re.I)


def docs_from(sl):
    parts = np.split(sl, np.where(sl == 2)[0] + 1)
    for p in parts[1:-1] or parts:      # drop the cut-off first/last document
        p = p[p != 2]
        if len(p):
            yield TOK.decode([int(x) for x in p])


def clean(t, lang):
    t = URL.sub(" ", t)
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t).strip()
    if len(t) > 900:                    # cut at a sentence boundary near 900 chars
        cut = max(t.rfind(". ", 400, 900), t.rfind("? ", 400, 900), t.rfind("! ", 400, 900), t.rfind("\n", 400, 900))
        t = t[: cut + 1] if cut > 0 else t[:900]
    return t.strip()


def ok(t, lang, bucket):
    if not (70 <= len(t) <= 950):
        return False
    w = [x.lower() for x in WORD.findall(t)]
    if len(w) < 20 or BOILER.search(t):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters / len(t) < 0.62 or sum(c.isdigit() for c in t) / len(t) > 0.12:
        return False
    if len(set(w)) / len(w) < 0.45:     # repetitive
        return False
    en = sum(x in EN for x in w) / len(w)
    fr = sum(x in FR for x in w) / len(w)
    if lang not in ("pcm", "afr") and en > 0.15:
        return False
    if lang not in ("wol", "fon", "kin", "ewe", "lin", "run", "bam") and fr > 0.22:
        return False
    if t.count("?") > 6 or t.count("!") > 8:
        return False
    return True


def main(out, target, langs):
    os.makedirs(out, exist_ok=True)
    for lang in langs:
        os.environ["N_SLICES"] = str(max(300, target // 3))
        ex.N = int(os.environ["N_SLICES"])
        tmp = f"{out}/_slices"
        ex.main(tmp, [lang])
        z = np.load(f"{tmp}/{lang}.npz")
        tok, meta = z["tok"], json.loads(str(z["meta"]))
        off, seen, by = 0, set(), {}
        for m in meta:
            sl = tok[off: off + m["n"]]
            off += m["n"]
            for d in docs_from(sl):
                d = clean(d, lang)
                if not ok(d, lang, m["bucket"]):
                    continue
                h = hashlib.md5(" ".join(re.findall(r"\w+", d.lower())[:12]).encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                by.setdefault(m["bucket"], []).append({"seed_id": h[:12], "lang": lang, "track": "B", "genre": GENRE[m["bucket"]], "bucket": m["bucket"],
                                                      "text": d, "src": m["src"], "license": "internal-corpus"})
        # drop the noisy 'pairs' bucket when other buckets carry enough
        if sum(len(v) for b, v in by.items() if b != "pairs") >= 0.6 * target:
            by.pop("pairs", None)
        rnd = random.Random(lang)
        # stratify: equal-ish per bucket (cap one source at 25% of the language), fill the rest by availability
        rows, srcn = [], {}
        buckets = {b: rnd.sample(v, len(v)) for b, v in by.items()}
        quota = {b: max(1, int(target / max(1, len(buckets)) * 1.2)) for b in buckets}
        for b, v in buckets.items():
            for r in v[: quota[b]]:
                if srcn.get(r["src"], 0) >= max(25, int(0.25 * target)):
                    continue
                srcn[r["src"]] = srcn.get(r["src"], 0) + 1
                rows.append(r)
        rnd.shuffle(rows)
        rows = rows[:target]
        with open(f"{out}/{lang}.vast.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        import collections
        print(lang, len(rows), dict(collections.Counter(r["bucket"] for r in rows)), "sources", len(srcn), flush=True)
        os.remove(f"{tmp}/{lang}.npz")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
