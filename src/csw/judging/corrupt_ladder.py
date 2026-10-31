"""Graded synthetic corruptions ("ladders") so the quality estimator and the judge calibration see the WHOLE grade range, per rubric dimension, not only 'fine' and 'destroyed'.

Every operator damages ONE rubric dimension at three intensities (level 1 = mild, 2 = moderate, 3 = severe) and needs no knowledge of the language: it only moves, drops, duplicates,
strips or injects material that already exists in the rows (the row's own other fields, another row of the same language, English words of the source, a lexicon of another language).
A row + operator gives an ordered chain  original > level 1 > level 2 > level 3  for the damaged dimension, i.e. free pairwise-ranking supervision with known order.
The expected grade is NOT assumed: run the chains through the judges and measure how each judge's grade moves; that is also the judge calibration (a judge that scores level 1 and level 3
the same cannot see that dimension).

    from csw.judging.corrupt_ladder import OPS, make_chain
    chain = make_chain(row, "H5_under", rnd, ctx)      # -> [{'level':1,'row':{...},'meta':{...}}, ...]  (levels that do not apply are skipped)
ctx = {"pool_by_lang": {lang: [rows]}, "lexicon": Lexicon|None, "foreign_other": {lang: [tokens of another language]}}
"""
from __future__ import annotations

import copy
import random
import re
import unicodedata

SENT = re.compile(r"(?<=[.!?።؟…])\s+")
WORD = re.compile(r"\S+")
SPECIAL_MAP = str.maketrans({"ɛ": "e", "Ɛ": "E", "ɔ": "o", "Ɔ": "O", "ɓ": "b", "Ɓ": "B", "ɗ": "d", "Ɗ": "D", "ƙ": "k", "Ƙ": "K", "ƴ": "y", "ŋ": "n", "Ŋ": "N", "ɲ": "n",
                             "ẹ": "e", "ọ": "o", "ṣ": "s", "ị": "i", "ụ": "u", "ṅ": "n", "ɖ": "d", "ʋ": "v", "ɣ": "g", "ƒ": "f"})
LEVELS = (1, 2, 3)


def sents(t: str) -> list[str]:
    return [s for s in SENT.split(t.strip()) if s]


def words(t: str) -> list[str]:
    return WORD.findall(t)


def join(ss: list[str]) -> str:
    return " ".join(ss)


# ---------------------------------------------------------------- field-level primitives (text -> text)
def p_drop_sentences(t, frac, rnd):
    ss = sents(t)
    if len(ss) < 2:
        return None
    k = max(1, round(len(ss) * frac)); k = min(k, len(ss) - 1)
    keep = sorted(rnd.sample(range(len(ss)), len(ss) - k))
    return join([ss[i] for i in keep])


def p_truncate(t, keep_frac, rnd):
    w = words(t)
    if len(w) < 8:
        return None
    return " ".join(w[: max(4, int(len(w) * keep_frac))])


def p_word_swaps(t, rate, rnd):
    w = words(t)
    if len(w) < 6:
        return None
    n = max(1, int(len(w) * rate / 2))
    for _ in range(n):
        i = rnd.randrange(len(w) - 1); w[i], w[i + 1] = w[i + 1], w[i]
    return " ".join(w)


def p_duplicate_words(t, rate, rnd):
    w = words(t)
    if len(w) < 6:
        return None
    out = []
    for x in w:
        out.append(x)
        if rnd.random() < rate:
            out.append(x)
    return " ".join(out)


def p_strip_marks(t, frac, rnd):
    chars = list(t); idx = [i for i, c in enumerate(chars) if c.translate(SPECIAL_MAP) != c or any(unicodedata.combining(x) for x in unicodedata.normalize("NFD", c))]
    if len(idx) < 4:
        return None
    pick = set(rnd.sample(idx, max(1, int(len(idx) * frac))))
    for i in pick:
        c = chars[i].translate(SPECIAL_MAP)
        chars[i] = "".join(x for x in unicodedata.normalize("NFD", c) if not unicodedata.combining(x))
    return unicodedata.normalize("NFC", "".join(chars))


def p_replace_sentences(t, donor, frac, rnd):
    """replace a fraction of the sentences of t with sentences of donor (another row / another field)"""
    ss, ds = sents(t), sents(donor)
    if len(ss) < 2 or not ds:
        return None
    k = max(1, round(len(ss) * frac)); idx = rnd.sample(range(len(ss)), min(k, len(ss)))
    for i in idx:
        ss[i] = rnd.choice(ds)
    return join(ss)


def p_inject_words(t, tokens, rate, rnd):
    w = words(t)
    if len(w) < 6 or not tokens:
        return None
    k = max(1, int(len(w) * rate)); idx = rnd.sample(range(len(w)), min(k, len(w)))
    for i in idx:
        w[i] = rnd.choice(tokens)
    return " ".join(w)


def p_insert_words(t, tokens, n, rnd):
    w = words(t)
    if len(w) < 6 or not tokens:
        return None
    for _ in range(n):
        w.insert(rnd.randrange(1, len(w)), rnd.choice(tokens))
    return " ".join(w)


def eng_tokens(row):
    return [x.strip(".,;:!?\"'()") for x in words(row["eng"]) if len(x.strip(".,;:!?\"'()")) >= 3]


# ---------------------------------------------------------------- operators:  (dimension, field, builder(row, level, rnd, ctx) -> new text or None)
def other_row(row, ctx, rnd):
    pool = [r for r in ctx.get("pool_by_lang", {}).get(row["lang"], []) if r["eng"] != row["eng"]]
    return rnd.choice(pool) if pool else None


def op_number(row, lv, rnd, ctx):               # eng changes, translations stay: the translations now disagree with the source
    from csw.judging.build_judge_batches import change_number
    e = row["eng"]
    for _ in range(lv):
        n = change_number(e, rnd)
        if not n:
            return None
        e = n
    r = copy.deepcopy(row); r["eng"] = e
    return r, "eng"


def op_negation(row, lv, rnd, ctx):
    from csw.judging.build_judge_batches import flip_negation
    e = row["eng"]
    for _ in range(lv):
        n = flip_negation(e, rnd)
        if not n:
            return None
        e = n
    r = copy.deepcopy(row); r["eng"] = e
    return r, "eng"


def field_op(field, fn):
    def run(row, lv, rnd, ctx):
        new = fn(row, lv, rnd, ctx)
        if not new or new == row[field]:
            return None
        r = copy.deepcopy(row); r[field] = new
        return r, field
    return run


def sub(field, builder):
    return field_op(field, lambda row, lv, rnd, ctx: builder(row, lv, rnd, ctx))


FR3 = (0.34, 0.67, 1.0)
OPS: dict[str, tuple[str, str, callable]] = {
    # name: (rubric dimension, field(s) damaged, operator)
    "M1_number": ("M1", "m_text", op_number),
    "M1_negation": ("M1", "m_text", op_negation),
    "M1_omission": ("M1", "m_text", sub("m_text", lambda r, lv, rnd, c: p_drop_sentences(r["m_text"], (0.25, 0.5, 0.75)[lv - 1], rnd))),
    "M1_addition": ("M1", "m_text", sub("m_text", lambda r, lv, rnd, c: (lambda o: join(sents(r["m_text"]) + rnd.sample(sents(o["m_text"]), min(lv, len(sents(o["m_text"]))))) if o else None)(other_row(r, c, rnd)))),
    "M2_wordorder": ("M2", "m_text", sub("m_text", lambda r, lv, rnd, c: p_word_swaps(r["m_text"], (0.10, 0.25, 0.5)[lv - 1], rnd))),
    "M2_repeat": ("M2", "m_text", sub("m_text", lambda r, lv, rnd, c: p_duplicate_words(r["m_text"], (0.05, 0.12, 0.25)[lv - 1], rnd))),
    "M3_diacritics": ("M3", "m_text", sub("m_text", lambda r, lv, rnd, c: p_strip_marks(r["m_text"], (0.25, 0.6, 1.0)[lv - 1], rnd))),
    "M4_otherlang": ("M4", "m_text", sub("m_text", lambda r, lv, rnd, c: (lambda o: p_replace_sentences(r["m_text"], o["m_text"], FR3[lv - 1], rnd) if o else None)(
        rnd.choice([x for l, v in c.get("pool_by_lang", {}).items() if l != r["lang"] and l not in ("pcm", "gpe", "wes") for x in v[:50]] or [None])))),
    "M4_english": ("M4", "m_text", sub("m_text", lambda r, lv, rnd, c: p_inject_words(r["m_text"], eng_tokens(r), (0.08, 0.2, 0.4)[lv - 1], rnd))),
    "M_truncate": ("M1", "m_text", sub("m_text", lambda r, lv, rnd, c: p_truncate(r["m_text"], (0.8, 0.55, 0.3)[lv - 1], rnd))),
    "H1_foreign_sentences": ("H1", "hm", sub("hm", lambda r, lv, rnd, c: (lambda o: p_replace_sentences(r["hm"], o["hm"], FR3[lv - 1], rnd) if o else None)(other_row(r, c, rnd)))),
    "H2_wordorder": ("H2", "hm", sub("hm", lambda r, lv, rnd, c: p_word_swaps(r["hm"], (0.10, 0.25, 0.5)[lv - 1], rnd))),
    "H3_switchpoint": ("H3", "hm", sub("hm", lambda r, lv, rnd, c: p_insert_words(r["hm"], ["the", "of", "and", "is", "to", "that"] if r["scenario"]["switch"] == "English" else eng_tokens(r), (1, 3, 6)[lv - 1], rnd))),
    "H4_thirdlang": ("H4", "hm", sub("hm", lambda r, lv, rnd, c: p_inject_words(r["hm"], c.get("foreign_other", {}).get(r["lang"], []), (0.04, 0.10, 0.20)[lv - 1], rnd))),
    "H5_undermix": ("H5", "hm", sub("hm", lambda r, lv, rnd, c: p_replace_sentences(r["hm"], r["m_text"], FR3[lv - 1], rnd) if len(sents(r["hm"])) == len(sents(r["m_text"])) else None)),
    "H5_overmix": ("H5", "hm", sub("hm", lambda r, lv, rnd, c: p_inject_words(r["hm"], eng_tokens(r), (0.12, 0.25, 0.45)[lv - 1], rnd))),
    "H6_integration": ("H6", "hm", sub("hm", lambda r, lv, rnd, c: (lambda t: t if t != r["hm"] else None)(re.sub(r"\b([A-Za-zÀ-ÿ]{1,4})-(?=[A-Za-z])", lambda m: "" if rnd.random() < (0.5, 0.8, 1.0)[lv - 1] else m.group(0), r["hm"])))),
    "H7_diacritics": ("H7", "hm", sub("hm", lambda r, lv, rnd, c: p_strip_marks(r["hm"], (0.25, 0.6, 1.0)[lv - 1], rnd))),
    "H_truncate": ("H1", "hm", sub("hm", lambda r, lv, rnd, c: p_truncate(r["hm"], (0.8, 0.55, 0.3)[lv - 1], rnd))),
    "O1_foreign_sentences": ("O1", "ood", sub("ood", lambda r, lv, rnd, c: (lambda o: p_replace_sentences(r["ood"], o["ood"], FR3[lv - 1], rnd) if o else None)(other_row(r, c, rnd)))),
    "O2_idiom_literal": ("O2", "ood", sub("ood", lambda r, lv, rnd, c: _idiom_to_gloss(r, lv, rnd))),
    "O2_idiom_scramble": ("O2", "ood", sub("ood", lambda r, lv, rnd, c: _idiom_scramble(r, lv, rnd))),
    "O5_english": ("O5", "ood", sub("ood", lambda r, lv, rnd, c: p_inject_words(r["ood"], eng_tokens(r), (0.08, 0.2, 0.4)[lv - 1], rnd))),
    "O6_diacritics": ("O6", "ood", sub("ood", lambda r, lv, rnd, c: p_strip_marks(r["ood"], (0.25, 0.6, 1.0)[lv - 1], rnd))),
    "O7_flat": ("O7", "ood", sub("ood", lambda r, lv, rnd, c: p_replace_sentences(r["ood"], r["m_text"], FR3[lv - 1], rnd) if len(sents(r["ood"])) == len(sents(r["m_text"])) else None)),
    "O_truncate": ("O1", "ood", sub("ood", lambda r, lv, rnd, c: p_truncate(r["ood"], (0.8, 0.55, 0.3)[lv - 1], rnd))),
}


def _devices(r):
    out = []
    for d in r.get("devices") or []:
        if isinstance(d, str):
            m = re.match(r"^(.*?)\s*[\(（](.*)[\)）]\s*$", d, re.S)
            out.append((m.group(1).strip(), m.group(2).strip()) if m else (d.strip(), ""))
    return [(t, g) for t, g in out if t and t in r["ood"]]


def _idiom_to_gloss(r, lv, rnd):
    ds = [(t, g) for t, g in _devices(r) if g]
    if not ds:
        return None
    k = {1: 1, 2: max(1, len(ds) // 2), 3: len(ds)}[lv]
    t = r["ood"]
    for term, gloss in rnd.sample(ds, min(k, len(ds))):
        gloss = re.sub(r"^(idiom|metaphor|ideophone|proverb|slang)[^:]*:\s*", "", gloss, flags=re.I)
        t = t.replace(term, gloss, 1)
    return t


def _idiom_scramble(r, lv, rnd):
    ds = _devices(r)
    if not ds:
        return None
    k = {1: 1, 2: max(1, len(ds) // 2), 3: len(ds)}[lv]
    t = r["ood"]
    for term, _ in rnd.sample(ds, min(k, len(ds))):
        w = term.split()
        if len(w) < 2:
            continue
        w2 = w[:]
        for _ in range(5):
            rnd.shuffle(w2)
            if w2 != w:
                break
        t = t.replace(term, " ".join(w2), 1)
    return t


def make_chain(row: dict, op: str, rnd: random.Random, ctx: dict) -> list[dict]:
    dim, field, fn = OPS[op]
    out = []
    for lv in LEVELS:
        try:
            res = fn(row, lv, rnd, ctx)
        except Exception:
            res = None
        if not res:
            continue
        new, f = res
        if out and out[-1]["row"][f] == new[f]:
            continue                                 # two levels gave the same text (e.g. the row has only one idiom): keep the chain strictly graded
        out.append({"level": lv, "row": new, "meta": {"op": op, "dim": dim, "field": f, "level": lv, "base_id": row.get("id")}})
    return out
