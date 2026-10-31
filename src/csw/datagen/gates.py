"""Deterministic gates run before (and after) the LLM judge. Cheap, language-independent, no models."""
from __future__ import annotations

import os
import re
import unicodedata

EN = set("the a an and or but of to in on at for with from by is are was were be been being have has had do does did not no yes this that these those it its i you he she we they me him her us them my your his our their what which who whom when where why how if then than so very just also can could will would shall should may might must about into over after before again more most some any all each every too only own same such there here up down out off as because while still even like going get got gonna wanna really actually please thanks okay".split())
FR = set("le la les un une des et ou mais de du au aux en dans sur avec pour par est sont était être avoir ai as a avons avez ont ne pas non oui ce cet cette ces il elle nous vous ils elles je tu me te se mon ma mes ton ta tes son sa ses notre votre leur qui que quoi dont où quand comment pourquoi si alors très aussi plus moins bien donc voilà quand même".split())
ED_OK = {"wicked", "naked", "sacred", "beloved", "kindred", "hundred", "wretched", "crooked", "rugged", "learned"}   # real adjectives that Pidgin keeps
PIDGIN_BAD = set("is are was were i'm it's he's she's we're they're you're isn't aren't wasn't has have had will would can't won't doesn't didn't peacefully settle kindly unfortunately however therefore regarding".split())
FORMAL_PAST = set("understood went left saw came made took gave knew thought said told felt became had was were been being seek unfortunately however therefore final".split())
PIDGIN = set("wey dey don abeg wahala sef sha oga una unu naa nawa sabi comot wetin pikin waka".split())
PT = set("o a os as um uma uns umas e ou mas de do da dos das no na nos nas em com para por que não sim eu tu ele ela nós vocês eles elas meu minha seu sua isso isto aquilo muito mais menos também então porque quando como onde está estão foi ser ter tem já".split())
SWITCH_WORDS = {"English": EN, "French": FR, "Portuguese": PT}
SCRIPTS = {"arabic": re.compile(r"[؀-ۿ]"), "geez": re.compile(r"[ሀ-፿]"), "cjk": re.compile(r"[一-鿿぀-ヿ]"),
           "cyrillic": re.compile(r"[Ѐ-ӿ]")}
WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.U)


def words(t: str) -> list[str]:
    return [w.lower() for w in WORD.findall(unicodedata.normalize("NFC", t))]


def ratio(ws: list[str], vocab: set[str]) -> float:
    return sum(w in vocab for w in ws) / max(1, len(ws))


def sentences(t: str) -> int:
    return max(1, len(re.findall(r"[.!?…።؟]+(?:\s|$)", t)))


EN_LEAK_EXEMPT_BASE = {"pcm", "gpe", "wes", "afr"}
EN_LEAK_MAX_SHARE = float(os.environ.get("CS_EN_LEAK_MAX", 0.06))          # hard-fail when leaked English words are more than this share of hm words (and at least 2)


def en_leak(row: dict) -> tuple[int, float] | None:
    """(count, share of hm words) of hm words (4+ letters) copied verbatim from the English source and absent from m_text, when the pinned switch language is not English.
    None when the check does not apply: English pin, Pidgin or Sheng pin (English-derived by nature), Pidgin/Afrikaans base language."""
    sw = (row.get("scenario") or {}).get("switch") or ""
    if sw == "English" or "Pidgin" in sw or "Sheng" in sw or row.get("lang") in EN_LEAK_EXEMPT_BASE:
        return None
    ew = {w for w in words(row["eng"]) if len(w) >= 4}; mw = set(words(row["m_text"])); hw = words(row["hm"])
    n = sum(1 for w in hw if len(w) >= 4 and w in ew and w not in mw)
    return n, round(n / max(1, len(hw)), 3)


def check(row: dict, prof) -> dict:
    """Returns {'fail': [...], 'warn': [...], 'metrics': {...}}. `fail` rows are dropped; `warn` lowers the score."""
    fail, warn, m = [], [], {}
    for f in ("eng", "m_text", "hm", "ood"):
        v = row.get(f) or ""
        if not v.strip():
            fail.append(f"empty_{f}")
        elif unicodedata.normalize("NFC", v) != v:
            row[f] = unicodedata.normalize("NFC", v)
    if fail:
        return {"fail": fail, "warn": warn, "metrics": m}
    sc = row["scenario"]
    if sc.get("length_spec") == "v2" and sc.get("n_sent"):                  # requested length is enforced (v2 prompts only: older rows had other bands)
        n, want = sentences(row["eng"]), int(sc["n_sent"])
        tol = 1 if want <= 6 else (2 if want <= 12 else 3)
        m["eng_sentences"], m["eng_sentences_wanted"] = n, want
        if abs(n - want) > 2 * tol:
            fail.append("length_violation")
        elif abs(n - want) > tol:
            warn.append("length_off")
    sw = row["scenario"]["switch"]
    vocab = SWITCH_WORDS.get(sw, set())
    wh, wo, wm, we = words(row["hm"]), words(row["ood"]), words(row["m_text"]), words(row["eng"])
    m["sw_hm"] = round(ratio(wh, vocab), 3)            # share of hm words that are common English/French function words
    m["sw_ood"] = round(ratio(wo, vocab | EN), 3)
    m["sw_m"] = round(ratio(wm, vocab | EN), 3)
    m["len_ratio_m"] = round(len(wm) / max(1, len(we)), 2)
    m["len_ratio_ood"] = round(len(wo) / max(1, len(we)), 2)
    m["uniq_hm"] = round(len(set(wh)) / max(1, len(wh)), 2)
    m["hm_eq_m"] = row["hm"].strip() == row["m_text"].strip()
    m["ood_eq_m"] = row["ood"].strip() == row["m_text"].strip()
    if m["hm_eq_m"]:
        fail.append("hm_equals_m_text")
    if m["ood_eq_m"]:
        fail.append("ood_equals_m_text")
    if row["m_text"].strip().lower() == row["eng"].strip().lower() and prof.code not in ("pcm", "gpe", "wes"):   # English IS the written form of the Pidgins
        fail.append("m_text_is_english")
    if m["sw_ood"] > 0.40 and prof.code != "pcm":        # ood may code-switch; only a mostly-English ood is dropped
        fail.append("ood_english_leak")
    elif m["sw_ood"] > 0.20 and prof.code != "pcm":
        warn.append("ood_english_heavy")
    en_share = ratio(wh, EN)
    m["en_hm"] = round(en_share, 3)
    if sw != "English" and en_share > 0.06 and prof.code not in ("pcm", "afr"):
        warn.append("english_when_not_pinned")        # judge models miss this; the pin is a hard rule of the prompt
    leak = en_leak(row)
    if leak is not None:
        m["en_leak"], m["en_leak_share"] = leak
        if leak[0] >= 2 and leak[1] > EN_LEAK_MAX_SHARE:
            fail.append("english_leak_unpinned")
    # Any Pidgin (base language or pinned): formal English is a tell of non-native Pidgin. Counted over hm + ood.
    pidgin_present = prof.code in ("pcm", "gpe", "wes") or "Pidgin" in sw
    if pidgin_present:
        toks = wh + wo
        bad = sum(1 for w in toks if w in FORMAL_PAST or w in PIDGIN_BAD or (len(w) > 5 and w.endswith("ed") and w.isascii() and w not in ED_OK))
        m["pidgin_formal"] = bad
        if bad >= 3:
            fail.append("pidgin_formal_english")
        elif bad >= 1:
            warn.append("pidgin_formal_english")
    # diacritic languages inserted into hm/ood: the pinned language's own diacritics must appear if it is one that needs them
    pin_need = {"Yoruba": r"[ẹọṣàáèéìíòóùúńǹ]", "Igbo": r"[ịọụṅ]", "Ewe": r"[ɖɛƒɣɔŋʋ]", "Lingala": r"[ɛɔáéíóú]"}.get(sw)
    if pin_need and prof.code != sw.lower()[:3]:
        if not re.search(pin_need, row["hm"]) and sum(c.isalpha() for c in row["hm"]) > 60:
            warn.append("pinned_lang_missing_diacritics")
    pid = ratio(wh, PIDGIN)
    m["pidgin_hm"] = round(pid, 3)
    if pid > 0.03 and "Pidgin" not in sw and prof.code not in ("pcm", "gpe", "wes"):
        warn.append("pidgin_leak_in_hm")
    if sw in SWITCH_WORDS and m["sw_hm"] < 0.02 and prof.code != "pcm":
        warn.append("hm_not_mixed")
    if m["sw_hm"] > 0.55:
        warn.append("hm_over_mixed")
    if m["sw_m"] > 0.08 and prof.code != "pcm":
        warn.append("m_text_leak")
    if not 0.4 <= m["len_ratio_m"] <= 3.0:
        warn.append("m_len_ratio")
    if m["uniq_hm"] < 0.45 and len(wh) > 20:
        warn.append("hm_repetitive")
    for name, rx in SCRIPTS.items():
        for f in ("m_text", "hm", "ood"):
            if name == "geez" and prof.code in ("amh", "tir"):
                continue
            ok_hm = f == "hm" and ((name == "arabic" and sw == "Arabic") or (name == "geez" and sw == "Amharic"))
            if rx.search(row[f]) and not ok_hm:
                fail.append(f"{f}_{name}_script")
    # diacritic presence: languages that require marks must show a minimum density in ood and hm
    need = {"yor": r"[ẹọṣàáèéìíòóùúńǹ]", "ibo": r"[ịọụṅ]", "ewe": r"[ɖɛƒɣɔŋʋ]", "twi": r"[ɛɔ]", "hau": r"[ɓɗƙƴ]", "fon": r"[ɖɔɛ]", "wol": r"[ñŋëóàé]", "lin": r"[ɛɔáéíóú]", "bam": r"[ɛɔɲŋ]", "ful": r"[ɓɗŋɲƴ]", "kik": r"[ĩũ]", "mos": r"[ɛɩʋɔãẽĩõũ]", "bin": r"[ẹọ]"}.get(prof.code)
    if need:
        for f in ("hm", "ood"):
            n_letters = sum(c.isalpha() for c in row[f])
            n_marks = len(re.findall(need, row[f], re.I))
            m[f"diac_{f}"] = round(n_marks / max(1, n_letters), 3)
            if n_letters > 40 and n_marks / n_letters < 0.02:
                warn.append(f"{f}_missing_diacritics")
    if re.search(r"\b[a-zA-Z]\.[a-zA-Z]\b|[a-z]\.\s?[o s]\.", row["hm"] + row["ood"]):
        warn.append("ascii_diacritic_hack")
    return {"fail": fail, "warn": warn, "metrics": m}
