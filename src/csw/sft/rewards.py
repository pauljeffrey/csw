"""Rule-based rewards for GRPO on the code-switcher (used next to the learned quality estimator).  No model needed; every part is in [0, 1].

    from csw.sft.rewards import reward
    r = reward(sample, completion, lex)          # sample: one RL prompt row (task, base, switch, english, must_keep, n_sent)
    r["total"], r["parts"]                       # parts are logged so a collapsed term is visible during training
    reward(sample, completion, lex, qe=0.83, w_qe=0.5)   # blends the QE score (0..1) with the rule score

The plan-first completion makes three consistency terms possible (the plan can be checked against both the source and the text):
  plan_precision - planned items that really occur in the written text            (a plan the text ignores earns nothing)
  plan_recall    - switch-language words in the text that the plan announced      (switching without announcing is penalised)
  plan_source    - hm with English pin: planned items must come from the English source (the words the model chose to KEEP)
plus format, language tags, switch ratio band, preservation of numbers / names, copy-vs-source, length, script / diacritics and pin purity.
A completion whose structure does not parse gets 0 (it cannot be scored on content).
"""
from __future__ import annotations

from csw.paths import DATA_ROOT
import re
import sys
from pathlib import Path

HERE = DATA_ROOT
from csw.datagen import gates                                   # noqa: E402
from csw.sft import fmt          # noqa: E402
from csw.sft import devices as dv          # noqa: E402

NEED_MARKS = {"yor": r"[ẹọṣàáèéìíòóùúńǹ]", "ibo": r"[ịọụṅ]", "ewe": r"[ɖɛƒɣɔŋʋ]", "twi": r"[ɛɔ]", "hau": r"[ɓɗƙƴ]", "fon": r"[ɖɔɛ]", "wol": r"[ñŋëóàé]",
              "lin": r"[ɛɔáéíóú]", "bam": r"[ɛɔɲŋ]", "ful": r"[ɓɗŋɲƴ]", "kik": r"[ĩũ]"}
PIDGINS = ("pcm", "gpe", "wes")
WEIGHTS = {
    "m_text": {"format": 0.15, "tags": 0.10, "preserve": 0.20, "copy": 0.15, "length": 0.15, "script": 0.25},
    "hm": {"format": 0.10, "tags": 0.10, "plan_precision": 0.12, "plan_recall": 0.12, "plan_source": 0.06, "ratio": 0.14, "preserve": 0.12, "length": 0.06,
           "script": 0.08, "purity": 0.10},
    "ood": {"format": 0.10, "tags": 0.10, "plan_precision": 0.20, "plan_size": 0.10, "preserve": 0.10, "copy": 0.12, "length": 0.08, "script": 0.10, "purity": 0.10},
}
NUM = re.compile(r"\d[\d,.]*")
NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")


def must_keep(english: str) -> list[str]:
    """Numbers and proper-name-like words of the English source that must survive in every rendering."""
    nums = [n.strip(",.") for n in NUM.findall(english)]
    names = []
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", english):
        if m.start() > 0 and not re.search(r"[.!?]\s*$", english[:m.start()]):
            names.append(m.group(0))
    return list(dict.fromkeys(nums + names))


def _ramp(x: float, lo0: float, lo1: float, hi1: float, hi0: float) -> float:
    """0 outside [lo0, hi0], 1 inside [lo1, hi1], linear in between."""
    if x <= lo0 or x >= hi0:
        return 0.0
    if x < lo1:
        return (x - lo0) / (lo1 - lo0)
    if x > hi1:
        return (hi0 - x) / (hi0 - hi1)
    return 1.0


def _in_text(item: str, canon_text: str) -> bool:
    c = dv._canon(item)
    return bool(c) and c in canon_text


def reward(sample: dict, completion: str, lex: dv.Lexicon | None = None, qe: float | None = None, w_qe: float = 0.5, weights: dict | None = None) -> dict:
    task, base, eng = sample["task"], sample["base"], sample["english"]
    sw = sample.get("switch")                                     # tag, hm only (None for m_text / ood)
    p = fmt.parse(task, base, completion)
    parts: dict[str, float] = {}
    if not p["ok"]:
        return {"total": 0.0, "parts": {"format": 0.0}, "parsed": p}
    text, devs = p["text"], p["devices"]
    parts["format"] = 1.0
    want = [base] if task == "m_text" else ([base, sw] if task == "hm" else None)
    if task == "ood":
        parts["tags"] = 1.0 if p["tags"] in ([base], [base, sw]) else 0.0     # the second tag is optional: written when the text really mixes
    else:
        parts["tags"] = 1.0 if p["tags"] == want else (0.5 if p["tags"][:1] == [base] else 0.0)
    canon = dv._canon(text)
    ew = gates.words(eng); tw = gates.words(text)
    keep = sample.get("must_keep") or must_keep(eng)
    parts["preserve"] = (sum(_in_text(k, canon) for k in keep) / len(keep)) if keep else 1.0
    ratio_len = len(tw) / max(1, len(ew))
    expect = (lex.len_ratio.get(base) if lex is not None else None) or 1.0          # words per English word is language specific (Zulu ~0.6, Pidgin ~1.1)
    parts["length"] = _ramp(ratio_len / expect, 0.45, 0.75, 1.5, 2.4)
    # script contamination and required diacritics
    bad = 0
    for name, rx in gates.SCRIPTS.items():
        if rx.search(text) and not (name == "geez" and base in ("amh", "tir")):
            bad += 1
    parts["script"] = 0.0 if bad else 1.0
    need = NEED_MARKS.get(base)
    letters = sum(c.isalpha() for c in text)
    if need and letters > 40 and task != "m_text":
        parts["script"] *= 1.0 if len(re.findall(need, text, re.I)) / letters >= 0.02 else 0.5
    if task in ("m_text", "ood") and base not in PIDGINS:
        own_en = [w for w in tw if len(w) >= 3 and w in set(ew) and w not in {k.lower() for k in keep}]
        copy = len(own_en) / max(1, len(tw))
        parts["copy"] = _ramp(copy, -1, 0, 0.12, 0.45)                 # a rendering that is mostly source words is a copy, not a translation
    elif task in ("m_text", "ood"):
        parts["copy"] = 1.0
    if task in ("hm", "ood"):
        rate_items = [d for d in devs]
        parts["plan_precision"] = (sum(_in_text(d, canon) for d in rate_items) / len(rate_items)) if rate_items else 0.0
    if task == "ood":
        n_sent = max(1, int(sample.get("n_sent") or 1))
        parts["plan_size"] = _ramp(len(devs), 0.5, 2, max(3, n_sent + 1), max(5, 2 * n_sent + 3))
        pure = 1.0
        if lex is not None and base not in PIDGINS:
            en_share = sum(w in gates.EN for w in tw) / max(1, len(tw))
            pure = _ramp(en_share, -1, 0, 0.12, 0.40)
        parts["purity"] = pure
    if task == "hm":
        sw_name = fmt.TAG_NAME.get(sw, "English")
        name_tokens = {k.lower() for k in keep}
        info = None
        if lex is not None:
            fake = {"lang": base, "scenario": {"switch": "English" if sw == "eng" else sw_name}, "hm": text, "m_text": ""}
            info = dv.hm_devices(fake, lex, max_devices=50, exclude=name_tokens)
        if info is not None:
            parts["ratio"] = _ramp(info["switch_ratio"], 0.02, 0.10, 0.40, 0.65)
            found = {dv._canon(x) for x in info["devices"]}
            planned = " ".join(dv._canon(d) for d in devs)
            tokens_found = [t for x in found for t in x.split()]
            parts["plan_recall"] = (sum(t in planned.split() for t in tokens_found) / len(tokens_found)) if tokens_found else (1.0 if not devs else 0.5)
        else:
            parts["ratio"] = parts["plan_recall"] = 0.5
        if sw == "eng":
            et = set(ew)
            parts["plan_source"] = (sum(all(t in et for t in dv._canon(d).split()) for d in devs) / len(devs)) if devs else 0.0
            parts["purity"] = 1.0
        else:
            en_share = sum(w in gates.EN for w in tw) / max(1, len(tw))
            parts["plan_source"] = 1.0
            parts["purity"] = _ramp(en_share, -1, 0, 0.05, 0.25)       # a pinned non-English switch must not be done in English
    w = weights or WEIGHTS[task]
    rule = sum(w[k] * parts.get(k, 0.0) for k in w) / sum(w.values())
    if parts["tags"] == 0.0:
        rule *= 0.5                       # the language tags are the first thing the policy commits to; a wrong one is never worth a good text
    if parts.get("copy", 1.0) == 0.0 or parts["length"] == 0.0 or parts["preserve"] < 0.2 and keep:
        rule *= 0.3                       # hard failures (a copy of the source, a truncation / run-on, lost names and numbers) cannot be bought back by easy terms
    total = rule if qe is None else (1 - w_qe) * rule + w_qe * float(qe)
    return {"total": round(total, 4), "rule": round(rule, 4), "parts": {k: round(v, 3) for k, v in parts.items()}, "parsed": p}
