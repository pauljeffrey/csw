"""Extract the PLAN ("devices") for each task from the generated rows, so the model can learn to plan before it writes.

hm  : the switch-language words / phrases that were put into the sentence (English words kept, or French / Arabic / Hausa ... words, whatever the pin was).
      There is no stored hm_devices field, so they are recovered from the text.  A token counts as a switch token when
        (a) it is not in the same row's m_text (the plain translation) - a word that was already there is not a switch, and
        (b) it is in the corpus-level 'foreign lexicon' of (base language, switch language): words that appear >=3 times in that pair's hm texts but at most 10% as
            often in the base language's m_text, OR (switch language English) it is a known English word of >=3 letters.
      Hyphenated prefix forms ("i-digital", "ku-maintain", "degree-ኡን") are split: the device is the switched stem.  Adjacent switched tokens merge into one phrase.
ood : the native idioms the generator reported (`devices`), minus the English gloss in brackets, kept only when they really occur in the ood text.
"""
from __future__ import annotations

import collections
import json
import re
import unicodedata
from pathlib import Path

WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.U)
SPLIT = re.compile(r"['’\-]")
STAR = re.compile(r"\*+")


def norm(t: str) -> str:
    return unicodedata.normalize("NFC", STAR.sub("", t))


def subtokens(w: str) -> list[str]:
    return [s for s in SPLIT.split(w) if s]


def tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD.finditer(norm(text))]


def key_tokens(text: str) -> set[str]:
    return {s.lower() for w, _, _ in tokens(text) for s in subtokens(w)}


class Lexicon:
    PIDGINS = ("pcm", "gpe", "wes")

    def __init__(self, foreign: dict[str, set[str]], english: set[str], base_known: dict[str, set[str]] | None = None, len_ratio: dict[str, float] | None = None):
        self.foreign = foreign          # "lang|Switch" -> set of switch-language tokens learned from the corpus
        self.english = english
        self.len_ratio = len_ratio or {}         # lang -> median (words in m_text) / (words in eng); agglutinative languages are well below 1
        self.base_known = base_known or {}      # Pidgin bases only: tokens seen >=2 times in that language's plain m_text (Pidgin is written mostly in English-looking words)

    @classmethod
    def build(cls, rows: list[dict], min_en_count: int = 2) -> "Lexicon":
        en = collections.Counter()
        m_cnt = collections.defaultdict(collections.Counter)
        hm_cnt = collections.defaultdict(collections.Counter)
        for r in rows:
            for s in key_tokens(r["eng"]):
                en[s] += 1
            m_cnt[r["lang"]].update(key_tokens(r["m_text"]))      # plain translation only: ood/hm also use switch words, so they would hide them
            hm_cnt[(r["lang"], r["scenario"]["switch"])].update(key_tokens(r["hm"]))
        foreign = {}
        for (lang, sw), c in hm_cnt.items():
            base = m_cnt[lang]
            foreign[f"{lang}|{sw}"] = {t for t, n in c.items() if n >= 3 and len(t) >= 3 and base.get(t, 0) <= 0.1 * n}
        english = {t for t, n in en.items() if n >= min_en_count and len(t) >= 3 and t.isascii()}
        known = {l: {t for t, n in m_cnt[l].items() if n >= 2} for l in cls.PIDGINS if l in m_cnt}
        import statistics
        rat = collections.defaultdict(list)
        for r in rows:
            rat[r["lang"]].append(len(tokens(r["m_text"])) / max(1, len(tokens(r["eng"]))))
        return cls(foreign, english, known, {l: round(statistics.median(v), 3) for l, v in rat.items()})

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({"foreign": {k: sorted(v) for k, v in self.foreign.items()}, "english": sorted(self.english), "base_known": {k: sorted(v) for k, v in self.base_known.items()}, "len_ratio": self.len_ratio}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: set(v) for k, v in d["foreign"].items()}, set(d["english"]), {k: set(v) for k, v in d.get("base_known", {}).items()}, d.get("len_ratio", {}))


def hm_devices(row: dict, lex: Lexicon, max_devices: int = 10, field: str = "hm", exclude: set | None = None) -> dict:
    """`field` = "hm" (the plan) or "ood" (used only to decide whether the ood text is mixed).  -> {'devices': [phrases in text order], 'switch_ratio': share of word tokens that are switch tokens, 'n_tokens': int}"""
    sw, lang = row["scenario"]["switch"], row["lang"]
    own = exclude if exclude is not None else key_tokens(row["m_text"])      # RL time has no m_text: the caller passes names/numbers of the English source instead
    foreign = lex.foreign.get(f"{lang}|{sw}", set())
    toks = tokens(row[field])
    flags = []                     # per token: surface of the switched stem(s) or None
    known = lex.base_known.get(lang) if lang in Lexicon.PIDGINS else None
    for w, a, b in toks:
        subs = subtokens(w)
        hit = []
        for s in subs:
            sl = s.lower()
            if sl in own:
                continue
            if sl in foreign or (sw == "English" and sl in lex.english and len(sl) >= 3):
                hit.append(s)
            elif known is not None and len(sl) >= 3 and sl not in known and sl not in lex.english and sw != "English":
                hit.append(s)       # Pidgin base: a word that is neither English-looking nor ever seen in Pidgin m_text is the switch language
        flags.append(" ".join(hit) if hit else None)
    spans, cur, prev_end = [], [], None
    text = norm(row[field])
    for (w, a, b), f in zip(toks, flags):
        if f is None or (prev_end is not None and text[prev_end:a].strip(" \u00a0")):     # a gap with punctuation ends the phrase
            if cur:
                spans.append(" ".join(cur)); cur = []
        if f is not None:
            cur.append(f)
        prev_end = b
    if cur:
        spans.append(" ".join(cur))
    seen, devs = set(), []
    for s in spans:
        k = s.lower()
        if k not in seen:
            seen.add(k); devs.append(s)
    return {"devices": devs[:max_devices], "switch_ratio": round(sum(f is not None for f in flags) / max(1, len(toks)), 3), "n_tokens": len(toks)}


def _canon(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", norm(t).lower())).strip()


GLOSS = re.compile(r"\s*[\(（].*$", re.S)


def ood_devices(row: dict, max_devices: int = 8) -> dict:
    """-> {'devices': idioms (gloss removed) that occur in the ood text, 'glossed': the same with gloss, 'coverage': found/reported}"""
    text = _canon(row.get("ood") or "")
    found, glossed, total = [], [], 0
    for d in row.get("devices") or []:
        if not isinstance(d, str):
            continue
        total += 1
        term = GLOSS.sub("", d).strip(" ;,.")
        c = _canon(term)
        if c and c in text and term not in found:
            found.append(term); glossed.append(d)
    return {"devices": found[:max_devices], "glossed": glossed[:max_devices], "coverage": round(len(found) / max(1, total), 2), "reported": total}
