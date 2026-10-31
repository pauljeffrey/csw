"""Deterministic reward rules for the csw RL.  Pure python, no model, no network.  Every rule returns (penalty >= 0, detail); the reward adds them with weights from csw_rl/config.py and caps the total
so rules never outweigh the evaluator.  A rule that cannot be computed for a language (no table) returns 0 and says so in `detail` -- never a guess.

parse(): completion -> pieces.  The draft (`<draft>..</draft>`) is cut out before anything is measured, so a long draft never inflates the length ratio of the final text; it gets its own cap (draft_penalty).
Format gate (hm / ood): exactly one `<dev> .. </dev>` block with >= 2 devices, a non-empty final text after it, the end-of-turn token present, and nothing after it.
"""
from __future__ import annotations
import json, re, unicodedata
from pathlib import Path

DRAFT = re.compile(r"<draft>(.*?)</draft>", re.S)
DEV = re.compile(r"<dev>(.*?)</dev>", re.S)
LANGTAGS = re.compile(r"^(?:\s*<[a-z]{3}>)+")
ANYTAG = re.compile(r"<[^>\s]{1,14}>")
NUM = re.compile(r"\d+(?:[.,]\d+)*")
EN_FUNC = set("the and of to is in that it for was with as on be are this but have from or by at an not they we you he she his her their our your what which who when there been has had will would can could should about into than them these those".split())


EOS_MARKERS = ("<end_of_turn>", "</s>")                 # the csw chat template ends a turn with <end_of_turn>; </s> is kept for the plain-text tests


def parse(comp: str, eos=EOS_MARKERS) -> dict:
    """-> dict(final, devices, draft, dev_ok, draft_closed, eos_ok, raw_tags_ok)"""
    raw = comp.replace("<pad>", "").rstrip(); marks = [m for m in eos if m in raw]; eos_ok = len(marks) == 1 and raw.endswith(marks[0]) and raw.count(marks[0]) == 1
    t = raw
    for m in eos: t = t.replace(m, " ")
    t = t.replace("<eos>", " ").strip()
    draft = None; draft_closed = True
    m = DRAFT.search(t)
    if m: draft = m.group(1).strip(); t = (t[:m.start()] + " " + t[m.end():]).strip()
    elif "<draft>" in t: draft_closed = False; draft = t.split("<draft>", 1)[1]; t = t.split("<draft>", 1)[0].strip()
    n_open, n_close = t.count("<dev>"), t.count("</dev>"); devs = []; final = t
    m = DEV.search(t)
    if m:
        devs = [re.sub(r"\s*\([^)]*\)\s*$", "", x).strip() for x in m.group(1).split(" | ") if x.strip()]
        final = (t[:m.start()] + " " + t[m.end():]).strip()
    final = ANYTAG.sub(" ", LANGTAGS.sub("", final)); final = " ".join(final.split())
    return {"final": final, "devices": devs, "draft": draft, "draft_closed": draft_closed, "eos_ok": eos_ok, "dev_ok": n_open == 1 and n_close == 1 and m is not None and len(devs) >= 2, "n_dev_open": n_open, "n_dev_close": n_close}


def format_penalty(p: dict, task: str) -> tuple[float, dict]:
    """1.0 = hard failure.  Applies to every task: end-of-turn; hm / ood: the dev block and a non-empty final text; an unclosed draft."""
    bad = []
    if not p["eos_ok"]: bad.append("eos")
    if not p["draft_closed"]: bad.append("draft_unclosed")
    if task != "eng_m":
        if not p["dev_ok"]: bad.append("dev_block")
        if not p["final"]: bad.append("empty_final")
    elif not p["final"]: bad.append("empty_final")
    return (1.0 if bad else 0.0), {"format_fail": bad}


def length_penalty(final: str, source: str, lo: float, hi: float) -> tuple[float, dict]:
    """Chars of the final text / chars of the source; 0 inside [lo, hi], grows linearly outside, capped at 1."""
    r = len(final) / max(1, len(source)); pen = 0.0 if lo <= r <= hi else min(1.0, (lo - r) / lo if r < lo else (r - hi) / hi)
    return pen, {"len_ratio": round(r, 3), "window": [lo, hi]}


def draft_penalty(draft: str | None, source: str, cap: float = 1.5) -> tuple[float, dict]:
    """The draft's content is free; only its size and repetition are punished."""
    if not draft: return 0.0, {"draft": False}
    r = len(draft) / max(1, len(source)); pen = 0.0 if r <= cap else min(1.0, (r - cap) / cap)
    return max(pen, loop_penalty(draft)[0]), {"draft": True, "draft_ratio": round(r, 3)}


LOOP_FLOORS = {2: 0.80, 3: 0.88, 4: 0.90}              # share of distinct n-grams; the human references are >= 0.887 / 0.96 / 0.986 at the 0.5th percentile (calibration on 6,031 test references)


def loop_penalty(text: str, floors: dict | None = None) -> tuple[float, dict]:
    """Repetition at 2-, 3- and 4-gram level (texts of >= 12 words): p_n = min(1, 2 (floor_n - u_n) / floor_n) for the share u_n of distinct n-grams below the floor; p = max over n."""
    w = text.split(); fl = floors or LOOP_FLOORS
    if len(w) < 12: return 0.0, {"loop": False}
    worst, det = 0.0, {}
    for n, f in fl.items():
        g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]; u = len(set(g)) / len(g); det[f"uniq{n}"] = round(u, 3)
        if u < f: worst = max(worst, min(1.0, 2 * (f - u) / f))
    det["loop"] = worst > 0; return worst, det


def number_penalty(final: str, source: str, number_words: set[str] | None = None) -> tuple[float, dict]:
    """Numerals of the source must survive.  Missing but replaced by a known number word (English or the language's) is fine; a changed numeral (different number present, source one absent) is a full penalty."""
    src = NUM.findall(source)
    if not src: return 0.0, {"numbers": 0}
    out = set(NUM.findall(final)); missing = [x for x in src if x not in out]
    if not missing: return 0.0, {"numbers": len(src), "missing": 0}
    words = set(re.findall(r"\w+", final.lower())); nw = number_words or set()
    covered = bool(nw & words)
    new_numbers = [x for x in out if x not in src]
    if new_numbers: return 1.0, {"numbers": len(src), "missing": len(missing), "changed": new_numbers[:3]}
    return (0.25 if covered else 0.6) * len(missing) / len(src), {"numbers": len(src), "missing": len(missing), "number_word_found": covered, "number_words_known": bool(nw)}


def english_leak(final: str, allow: bool, ref_share: float = 0.0, cap: float = 0.08) -> tuple[float, dict]:
    """Share of common English function words in the final text.  hm legitimately holds the switch language (English): only ood and eng_m are checked."""
    if allow: return 0.0, {"en_check": False}
    w = re.findall(r"\w+", final.lower()); s = sum(x in EN_FUNC for x in w) / max(1, len(w)); pen = 0.0 if s <= cap else min(1.0, (s - cap) / 0.2)
    return pen, {"en_share": round(s, 3)}


class FuncWords:
    """Function-word tables per language (gemma_mt/calibration/funcwords_v1.json: word -> counts).  Used to tell content words from stop words."""
    def __init__(self, path: str | Path, top: int = 60):
        d = json.load(open(path, encoding="utf-8")); self.t = {k: {w for w, *_ in v["top"][:top]} for k, v in d.items() if v.get("top")}

    def has(self, lang: str) -> bool: return lang in self.t

    def stop_share(self, lang: str, words: list[str]) -> float | None:
        if lang not in self.t or not words: return None
        return sum(w in self.t[lang] for w in words) / len(words)


def switch_penalty(final: str, devices: list[str], lo: float = 0.15, hi: float = 0.40, fw: FuncWords | None = None, switch_lang: str = "eng") -> tuple[float, dict]:
    """hm: the switched material (the devices as they occur in the text) should be 15-40% of the words, and should consist of content words rather than stop words."""
    nf = unicodedata.normalize("NFC", final.lower()); fw_ = re.findall(r"\w+", nf)
    if not fw_ or not devices: return 1.0, {"switch_share": 0.0, "devices_in_text": 0}
    hit = [d for d in devices if unicodedata.normalize("NFC", d.lower()) in nf]; words = sum(len(re.findall(r"\w+", d)) for d in hit); share = words / len(fw_)
    pen = 0.0 if lo <= share <= hi else min(1.0, (lo - share) / lo if share < lo else (share - hi) / hi)
    detail = {"switch_share": round(share, 3), "devices_in_text": len(hit), "devices": len(devices)}
    if fw is not None and fw.has(switch_lang) and hit:
        sw = [w for d in hit for w in re.findall(r"\w+", unicodedata.normalize("NFC", d.lower()))]; ss = fw.stop_share(switch_lang, sw)
        if ss is not None: detail["switch_stopword_share"] = round(ss, 3); pen = max(pen, min(1.0, max(0.0, ss - 0.5) * 2))      # mostly stop words switched -> penalty
    else: detail["stopword_table"] = False
    return pen, detail


EN_NUM = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first second third".split())
FR_FUNC = set("le la les de des du un une et en que qui dans pour sur avec par ce se ne pas au aux est sont mais ou où il elle ils elles nous vous je tu à ça cette ces son sa ses leur leurs plus très comme".split())


def strip_devices(final: str, devices: list[str]) -> str:
    """The base-language part of an hm text: the final text with the (case-insensitive) device spans removed."""
    t = final
    for d in sorted(set(devices), key=len, reverse=True):
        if d.strip(): t = re.sub(re.escape(d.strip()), " ", t, flags=re.I)
    return " ".join(t.split())


def base_leak(base_text: str, switch_lang: str, fw: "FuncWords | None" = None, cap: float = 0.08) -> tuple[float, dict]:
    """hm: outside the devices the text must be the base language.  Share of switch-language function words in the base part; 0 below `cap`, linear to 1 at cap + 0.2.  No table for the switch language -> 0 and says so."""
    w = re.findall(r"\w+", unicodedata.normalize("NFC", base_text.lower()))
    if len(w) < 6: return 0.0, {"base_leak": "short"}
    tab = (fw.t.get(switch_lang) if fw is not None else None) or (EN_FUNC if switch_lang == "eng" else FR_FUNC if switch_lang == "fra" else None)
    if not tab: return 0.0, {"base_leak": "no_table"}
    s = sum(x in tab for x in w) / len(w); return min(1.0, max(0.0, s - cap) / 0.2), {"base_leak": round(s, 3)}


CAP = re.compile(r"\b[A-Z][a-zA-Z\u00C0-\u024F'-]{2,}\b")
STARTER = re.compile(r"(?:^|[.!?:;\"\u201c\u201d()\[\]\n]\s*)([A-Za-z\u00C0-\u024F'-]+)")


def source_names(source: str) -> set[str]:
    """Names of an English source: capitalised words of >= 3 letters that never start a sentence / quotation (the first word of every segment is excluded, capitalised only by position) and never occur in lower case."""
    starters = {m.group(1) for m in STARTER.finditer(source)}; low = {w for w in re.findall(r"\b[a-z][a-z'-]+\b", source)}
    return {w for w in CAP.findall(source) if w not in starters and w.lower() not in low}


def name_penalty(final: str, source: str) -> tuple[float, dict]:
    """English-source tasks only: the names of the source must survive in the output (names are not translated).  p = share missing (NFC, case-insensitive); 0 when the source has none."""
    nm = source_names(source)
    if not nm: return 0.0, {"names": 0}
    f = unicodedata.normalize("NFC", final.lower()); miss = [x for x in nm if unicodedata.normalize("NFC", x.lower()) not in f]
    return len(miss) / len(nm), {"names": len(nm), "names_missing": len(miss)}


def device_use(final: str, devices: list[str]) -> tuple[float, dict]:
    """The planned devices (switches for hm, idioms for ood) must occur in the final text: p = share of the listed devices that do not."""
    if not devices: return 1.0, {"devices_listed": 0}
    nf = unicodedata.normalize("NFC", final.lower()); hit = sum(unicodedata.normalize("NFC", d.lower()) in nf for d in devices if d.strip())
    return 1.0 - hit / len(devices), {"devices_listed": len(devices), "devices_used": hit}


def base_presence(base_text: str, lang: str, fw: "FuncWords | None", floors: dict) -> tuple[float, dict]:
    """The text outside the devices must look like the base language: its share of that language's function words must not fall below the language's floor (0.6 x the 5th percentile of the human references).  p = (floor - s) / floor."""
    f = floors.get(lang)
    if f is None or fw is None or not fw.has(lang) or f <= 0: return 0.0, {"base_fw": "n/a"}
    w = re.findall(r"\w+", unicodedata.normalize("NFC", base_text.lower()))
    if len(w) < 8: return 0.0, {"base_fw": "short"}
    s = sum(x in fw.t[lang] for x in w) / len(w); return (0.0 if s >= f else min(1.0, (f - s) / f)), {"base_fw": round(s, 3)}
