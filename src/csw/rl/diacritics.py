"""Diacritic penalty (copy of the logic of gemma_mt/rl/diacritics.py, D-31; no import across packages).  units = combining marks (Mn) + special letters in the NFD text; density = units / NFD chars.
deficit = max(0, t - d) / t with t = the human P10 density of the language (data/diacritics_blocks.json, FLORES+ dev calibration of the translation project); deficit = 1 when a mark / letter / (letter, marks) cluster
outside the language's human inventory occurs, a mark is orphaned, or the density exceeds the human p99.  Only languages with `applies` (median human density >= 0.008) are judged; others -> 0 (and the detail says so)."""
from __future__ import annotations
import json, unicodedata
from pathlib import Path

SPECIAL = {"ɛ": "e", "ɔ": "o", "ɖ": "d", "ƒ": "f", "ɣ": "g", "ŋ": "n", "ʋ": "v", "ɓ": "b", "ɗ": "d", "ƙ": "k", "ɠ": "g", "ɲ": "n", "ə": "e", "ɩ": "i", "ʊ": "u", "ɨ": "i", "ʉ": "u", "ɱ": "m", "ʃ": "s", "ʒ": "z", "ɦ": "h",
           "Ɛ": "E", "Ɔ": "O", "Ɖ": "D", "Ƒ": "F", "Ɣ": "G", "Ŋ": "N", "Ʋ": "V", "Ɓ": "B", "Ɗ": "D", "Ƙ": "K", "Ɠ": "G", "Ɲ": "N", "Ə": "E", "Ɩ": "I", "Ʊ": "U"}
MIN_GUARD_CHARS = 25
_BLOCKS = None


def _nfd(t): return unicodedata.normalize("NFD", unicodedata.normalize("NFC", t))
def _hex(c): return f"{ord(c):04X}"
def is_mark(c): return unicodedata.category(c) == "Mn"
def units(t): return sum(1 for c in _nfd(t) if is_mark(c) or c in SPECIAL)
def density(t): return units(t) / max(len(_nfd(t)), 1)


def clusters(t):
    out, base, marks = [], "", []
    for c in _nfd(t):
        if is_mark(c): marks.append(c); continue
        if marks: out.append((base, tuple(marks)))
        base, marks = c, []
    if marks: out.append((base, tuple(marks)))
    return out


def blocks():
    global _BLOCKS
    if _BLOCKS is None: _BLOCKS = json.load(open(Path(__file__).parent / "data" / "diacritics_blocks.json", encoding="utf-8"))
    return _BLOCKS


_POOL = None


def pooled():
    """Pooled prior for languages without a calibration block: the union of all languages' allowed marks / letters (anti-hacking guards only, no density term)."""
    global _POOL
    if _POOL is None:
        m, l = {}, {}
        for e in blocks().values():
            al = e.get("allowed", {}); m.update(al.get("marks", {})); l.update(al.get("letters", {}))
        _POOL = {"applies": True, "allowed": {"marks": m, "letters": l, "clusters": {}}, "density": {"p99": 1.0}, "threshold": 0.0}
    return _POOL


def penalty(text: str, lang: str) -> tuple[float, dict]:
    e = blocks().get(lang)
    if e is None: e = pooled()
    if not e.get("applies"): return 0.0, {"diacritics": "n/a"}
    nfd = _nfd(text)
    if len(nfd) < MIN_GUARD_CHARS: return 0.0, {"diacritics": "short"}
    al = e.get("allowed", {}); okm, okl, okc = al.get("marks", {}), al.get("letters", {}), al.get("clusters", {})
    flags = []
    if any((is_mark(c) and _hex(c) not in okm) or (c in SPECIAL and _hex(c) not in okl) for c in nfd): flags.append("foreign")
    for b, ms in clusters(text):
        if not b or not (b.isalpha() or b in SPECIAL): flags.append("orphan"); break
        if okc and (b.lower() + "|" + ",".join(_hex(m) for m in ms)) not in okc: flags.append("unseen_cluster"); break
    d = density(text)
    if d > float(e["density"]["p99"]): flags.append("over")
    thr = float(e.get("threshold", 0.0)); det = {"diac_density": round(d, 4), "diac_thr": round(thr, 4)}
    if flags: det["diac_flags"] = flags; return 1.0, det
    return (max(0.0, thr - d) / thr if thr > 0 else 0.0), det
