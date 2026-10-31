"""Run on the GPU/CPU box (numpy only, read-only on CSW_RETOK_DIR): sample token slices per language and bucket.

  python -m csw.datagen.extract_slices OUTDIR LANG [LANG ...]      -> OUTDIR/<lang>.npz  (flat uint16 tokens + per-slice meta)

Each slice is a contiguous run of tokens from one <source>/<lang>.dedup.bin. Documents inside are separated by
eos (id 2); decoding and document splitting happen locally with the pretraining tokenizer, so nothing here needs
a tokenizer. Bucket = what kind of text the source name suggests; the next step maps buckets to genres.
"""
import glob, json, os, random, re, sys
import numpy as np

ROOT = os.environ.get("CSW_RETOK_DIR", "retok")   # directory with <source>/<lang>.dedup.bin token files
BUCKETS = [   # (bucket, whole-word keywords in the source path, share of slices) - first match wins, in this order
    ("proverb", {"proverb", "proverbs", "idiom", "idioms", "saying", "sayings"}, 0.05),
    ("story", {"story", "stories", "tinystories", "folk", "folktale", "fable", "glotstory", "glotstory2", "storybook", "cosmopedia"}, 0.14),
    ("religion", {"jw", "bible", "verse", "verses", "gospel", "hymn", "sermon", "quran", "watchtower", "jehovah"}, 0.05),
    ("qa", {"qa", "instruct", "instruction", "instructions", "ultrachat", "sft", "alpaca", "question", "questions", "advisory"}, 0.10),
    ("chat", {"chat", "conversation", "conversations", "dialog", "dialogue", "whatsapp", "youth", "friend", "friends", "roleplay", "simpra", "bot"}, 0.18),
    ("social", {"tweet", "tweets", "twitter", "sentiment", "sentiments", "emotion", "emotions", "comment", "comments", "naija", "naija2", "naija3", "nollysenti", "reddit", "forum", "afrisenti", "pidgin"}, 0.14),
    ("news", {"news", "bbc", "voa", "alffa", "asali", "headline", "headlines", "massivesumm", "summarization"}, 0.12),
    ("pairs", {"pairs", "parallel", "opus", "tatoeba", "flores", "ntrex", "smol"}, 0.07),
    ("general", None, 0.15),
]


def bucket_of(src):
    toks = set(re.split(r"[^a-z0-9]+", src.lower()))
    for b, kw, _ in BUCKETS:
        if kw is None or toks & kw:
            return b


SLICE = 6000          # tokens per slice (a handful of documents)
N = int(os.environ.get("N_SLICES", "400"))


ALIASES = {"ful": ["ful", "fuv"]}      # Fula appears under two tags in the corpus


def resolve(src, lang):
    for pat in (f"{ROOT}/{src}/{lang}.dedup.bin", f"{ROOT}/{src}*/{lang}.dedup.bin"):
        h = glob.glob(pat)
        if h:
            return h[0]
    return None


def main(out, langs):
    os.makedirs(out, exist_ok=True)
    table = json.load(open(f"{ROOT}/lang_table.json"))["per_lang_source"]
    for lang in langs:
        rnd = random.Random(lang)
        srcs = []
        for tag in ALIASES.get(lang, [lang]):
            for s, n in table.get(tag, {}).items():
                p = resolve(s, tag)
                if p and n > 20_000:
                    srcs.append((s, p, n))
            have = {p for _, p, _ in srcs}
            for p in glob.glob(f"{ROOT}/*/*/{tag}.dedup.bin") + glob.glob(f"{ROOT}/*/{tag}.dedup.bin"):   # sources missing from lang_table
                if p not in have and os.path.getsize(p) > 40_000:
                    srcs.append((os.path.relpath(os.path.dirname(p), ROOT), p, os.path.getsize(p) // 2))
        by = {}
        for s, p, n in srcs:
            by.setdefault(bucket_of(s), []).append((s, p, n))
        shares = {b: sh for b, _, sh in BUCKETS if b in by}
        tot = sum(shares.values())
        toks, meta = [], []
        for b, sh in shares.items():
            k = max(1, round(N * sh / tot))
            pool = by[b]
            w = [min(n, 5e8) ** 0.5 for _, _, n in pool]   # soften size so small sources still appear
            for _ in range(k):
                s, p, n = rnd.choices(pool, weights=w)[0]
                mm = np.memmap(p, dtype=np.uint16, mode="r")
                if len(mm) < SLICE * 2:
                    st = 0
                    sl = np.array(mm[: min(len(mm), SLICE)])
                else:
                    st = rnd.randrange(0, len(mm) - SLICE)
                    sl = np.array(mm[st: st + SLICE])
                toks.append(sl)
                meta.append({"src": s, "bucket": b, "off": int(st), "n": int(len(sl))})
        np.savez_compressed(f"{out}/{lang}.npz", tok=np.concatenate(toks) if toks else np.zeros(0, np.uint16),
                            meta=json.dumps(meta))
        print(lang, "sources", len(srcs), "buckets", {b: len(v) for b, v in by.items()}, "slices", len(toks), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
