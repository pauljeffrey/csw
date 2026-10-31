"""Batch-API request files for scoring generated rows with TWO judges (OpenAI gpt-4o and Gemini), plus a calibration set with known answers.  NOTHING IS SENT.

python -m csw.judging.build_judge_batches --rows DIR --gold ../exemplars/augmented_gold.jsonl --lexicon sft_v0/lexicon.json --out OUT [--per-lang 100] [--calib-per-kind 60]
        [--openai-model gpt-4o] [--gemini-model gemini-3.1-pro-preview] [--gemini-thinking low]

Stage 0 (calibration): gold rows (active languages) and corrupted copies - swap / truncate / wrong-language / number changed / negation flipped / clean - with the corruption recorded
    in meta.json.  Both judges score them; per language we then see what each one catches, before any larger spend.
Stage 1 (scoring): per active language `per-lang` generated rows, stratified 40% clean / 30% flagged / 30% random.  'flagged' = a gate warning, non-empty flags, min self-confidence <= 0.8
    or difficulty >= 4 (the model's own conf is almost flat, so it cannot define 'bad' alone; the rule gates and the flags can).
Judge output is scores only (cheap): one small JSON per row; corrections are a separate, later pass on the rows a judge marks as fixable.
Files per stage: openai.jsonl (Batch API: custom_id/method/url/body), gemini.jsonl (Gemini Batch API file input: key/request), meta.json (id -> language, stratum, corruption, set), cost.json.
"""
from __future__ import annotations

from csw.paths import DATA_ROOT
import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = DATA_ROOT
from csw.datagen import gates                                              # noqa: E402
from csw.datagen.profiles import P, TARGETS, HOLD                     # noqa: E402
from csw.sft import fmt    # noqa: E402
from csw.sft import devices as dv    # noqa: E402
from csw.sft import build_sft as bs    # noqa: E402

SYSTEM = """You are a strict expert reviewer of African-language text for a training dataset. Judge ONE sample. If you cannot read the language well enough to judge a field, say so; never guess and never reward fluent-looking text you cannot verify.

Fields:
- eng: formal, neutral English source.
- m_text: faithful standard translation of eng into the base language (correct meaning, numbers, names, negation; grammatical; real language; standard orthography and diacritics).
- hm: code-switched version of m_text. The base language leads; ONLY the pinned switch language supplies words or short phrases (roughly 15-40% of the words); base grammar kept; no third language. hm_plan lists the switch words the writer intended: check they really occur and belong to the pinned switch language.
- ood: informal native parlance of the same situation: idioms, proverbs, imagery, same facts, richer than hm. ood_idioms lists the idioms the writer intended (with a gloss): check they are real, natural, used correctly and fit the meaning; invented or misused idioms are a serious flaw.

Score m_text, hm and ood from 1 to 10 (10 = publishable by a native expert unchanged, 8 = good with minor slips, 6 = understandable but clear errors, 4 = mostly wrong or partly another language, 1-2 = wrong language, copied or meaningless). A changed number, name, fact or negation caps the field at 5. Return ONE JSON object only:
{"can_read": true, "language_level": "fluent|good|limited|poor", "scores": {"m_text": 0, "hm": 0, "ood": 0}, "meaning_ok": {"m_text": true, "hm": true, "ood": true}, "pin_ok": true, "idioms_ok": true, "problems": ["field: short problem"], "confidence": 0.0}
can_read=false or language_level poor means the target-language scores are not reliable. Keep problems to at most 4 short items."""


def user_card(r: dict, lex: dv.Lexicon | None) -> str:
    lang = r["lang"]; p = P[lang]
    sc = r.get("scenario") or {}
    plan = dv.hm_devices(r, lex)["devices"] if lex is not None else []
    ood_d = [d for d in (r.get("devices") or []) if isinstance(d, str)][:8]
    return (f"BASE LANGUAGE: {p.name} [{p.code}] | PINNED SWITCH LANGUAGE: {sc.get('switch') or r.get('switch')} | DIALECT: {sc.get('dialect') or 'standard'} | GENRE: {sc.get('genre') or r.get('genre')}\n"
            f"ORTHOGRAPHY NOTE: {p.orthography}\n"
            + json.dumps({"eng": r["eng"], "m_text": r["m_text"], "hm": r["hm"], "hm_plan": plan, "ood": r["ood"], "ood_idioms": ood_d}, ensure_ascii=False, indent=1))


def oa_request(cid: str, card: str, model: str) -> dict:
    return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": card}], "temperature": 0, "max_tokens": 450,
                     "response_format": {"type": "json_object"}}}


def gem_request(cid: str, card: str, thinking: str) -> dict:
    gc = {"temperature": 0, "max_output_tokens": 2500, "response_mime_type": "application/json"}
    if thinking:
        gc["thinking_config"] = {"thinking_level": thinking}           # Gemini 3.x; for 2.5 models use thinking_budget instead (check before submitting)
    return {"key": cid, "request": {"system_instruction": {"parts": [{"text": SYSTEM}]}, "contents": [{"role": "user", "parts": [{"text": card}]}], "generation_config": gc}}


# ---- corruptions of gold rows (the target-language text is never edited, so no language knowledge is needed) ----
NUM = {"one": "three", "two": "five", "three": "seven", "four": "nine", "five": "eight", "six": "two", "seven": "four", "eight": "three", "nine": "six", "ten": "twenty", "twenty": "ten",
       "hundred": "thousand", "thousand": "hundred", "first": "third", "second": "fifth", "weekly": "daily", "daily": "monthly"}
NEG = [(r"\bdo not\b", "do"), (r"\bdoes not\b", "does"), (r"\bdid not\b", "did"), (r"\bcannot\b", "can"), (r"\bmust not\b", "must"), (r"\bwill not\b", "will"), (r"\bnever\b", "always"),
       (r"\bis not\b", "is"), (r"\bare not\b", "are"), (r"\bwas not\b", "was"), (r"\bhave not\b", "have"), (r"\bhas not\b", "has")]


def change_number(e, rnd):
    ms = list(re.finditer(r"\b(" + "|".join(NUM) + r")\b", e, re.I))
    if ms:
        m = rnd.choice(ms); w = NUM[m.group().lower()]; return e[:m.start()] + (w.capitalize() if m.group()[0].isupper() else w) + e[m.end():]
    ds = list(re.finditer(r"\b\d+\b", e))
    if ds:
        m = rnd.choice(ds); return e[:m.start()] + str(int(m.group()) + rnd.choice([2, 3, 5])) + e[m.end():]
    return None


def flip_negation(e, rnd):
    hits = [(re.search(p, e, re.I), r) for p, r in NEG]; hits = [(m, r) for m, r in hits if m]
    if not hits:
        return None
    m, r = rnd.choice(hits); return e[:m.start()] + (r.capitalize() if m.group()[0].isupper() else r) + e[m.end():]


def truncate(t):
    w = t.split(); return " ".join(w[: max(4, int(len(w) * 0.45))])


def calibration_rows(gold: list[dict], per_kind: int, rnd: random.Random) -> list[tuple[dict, dict]]:
    gold = [g for g in gold if g["lang"] in P and g["lang"] in TARGETS and g["lang"] not in HOLD]
    by = collections.defaultdict(list)
    for g in gold:
        by[g["lang"]].append(g)
    out, used = [], set()
    for kind in ("clean", "swap", "truncate", "wronglang", "number", "negation"):
        pool = gold[:]; rnd.shuffle(pool); made = 0
        for g in pool:
            if made >= per_kind:
                break
            c = dict(g); note = {"corruption": kind}
            if kind == "swap":
                o = [x for x in by[g["lang"]] if x["eng"] != g["eng"]]
                if not o:
                    continue
                x = rnd.choice(o); c.update(m_text=x["m_text"], hm=x["hm"], ood=x["ood"], devices=x.get("devices"))
            elif kind == "truncate":
                c.update(m_text=truncate(g["m_text"]), hm=truncate(g["hm"]), ood=truncate(g["ood"]))
            elif kind == "wronglang":
                o = [x for x in gold if x["lang"] not in (g["lang"], "gpe", "wes", "pcm") and x["lang"] in ("swh", "hau", "zul", "som", "amh", "afr", "yor", "kin", "sna", "mlg")]
                if not o:
                    continue
                x = rnd.choice(o); c.update(m_text=x["m_text"], hm=x["hm"], ood=x["ood"], devices=x.get("devices")); note["from"] = x["lang"]
            elif kind == "number":
                e = change_number(g["eng"], rnd)
                if not e:
                    continue
                c["eng"] = e
            elif kind == "negation":
                e = flip_negation(g["eng"], rnd)
                if not e:
                    continue
                c["eng"] = e
            key = (kind, g["eng"] + g["lang"])
            if key in used:
                continue
            used.add(key); made += 1; out.append((c, note))
    return out


def stratum(r: dict) -> str:
    g = gates.check(r, P[r["lang"]])
    if g["fail"]:
        return "drop"
    conf = r.get("conf") if isinstance(r.get("conf"), dict) else {}
    cmin = min([v for v in conf.values() if isinstance(v, (int, float))] or [0.9])
    diff = r.get("difficulty") if isinstance(r.get("difficulty"), (int, float)) else 3
    if g["warn"] or r.get("flags") or cmin <= 0.8 or diff >= 4:
        return "flagged"
    return "clean" if cmin >= 0.9 and diff <= 3 else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); ap.add_argument("--lexicon")
    ap.add_argument("--gold", nargs="*", default=[str(HERE / "exemplars" / "augmented_gold.jsonl")])
    ap.add_argument("--per-lang", type=int, default=100); ap.add_argument("--calib-per-kind", type=int, default=60)
    ap.add_argument("--openai-model", default="gpt-4o"); ap.add_argument("--gemini-model", default="gemini-3.1-pro-preview"); ap.add_argument("--gemini-thinking", default="low")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)
    lex = dv.Lexicon.load(a.lexicon) if a.lexicon else None
    rows = bs.load_rows(a.rows, [g for g in a.gold if Path(g).exists()])
    gold = [r for r in rows if r["source"] == "gold"]; gen = [r for r in rows if r["source"] == "generated"]

    def write(stage: str, items: list[tuple[str, dict, dict]]):
        d = out / stage; d.mkdir(exist_ok=True)
        with open(d / "openai.jsonl", "w", encoding="utf-8") as fo, open(d / "gemini.jsonl", "w", encoding="utf-8") as fg:
            meta = {}
            for cid, r, note in items:
                card = user_card(r, lex)
                fo.write(json.dumps(oa_request(cid, card, a.openai_model), ensure_ascii=False) + "\n")
                fg.write(json.dumps(gem_request(cid, card, a.gemini_thinking), ensure_ascii=False) + "\n")
                meta[cid] = {"lang": r["lang"], "set": r["source"], "row_id": r["id"], **note}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
        return len(items)

    cal = []
    for c, note in calibration_rows(gold, a.calib_per_kind, rnd):
        cid = "cal-" + hashlib.sha1((note["corruption"] + c["eng"] + c["m_text"] + c["lang"]).encode()).hexdigest()[:12]
        cal.append((cid, c, note))
    n0 = write("stage0_calibration", cal)

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in gen:
        if r["lang"] in TARGETS and r["lang"] not in HOLD and r["lang"] in P:
            by[r["lang"]][stratum(r)].append(r)
    picked, counts = [], collections.Counter()
    for lang in sorted(by):
        s = by[lang]; want = {"clean": int(a.per_lang * 0.4), "flagged": int(a.per_lang * 0.3)}; got = []
        for k, n in want.items():
            pool = s.get(k, []) + (s.get("other", []) if k == "clean" else []); rnd.shuffle(pool)
            take = pool[:n]; got += [(r, k) for r in take]
        taken = {id(r) for r, _ in got}
        rest = [r for k, v in s.items() if k != "drop" for r in v if id(r) not in taken]; rnd.shuffle(rest)
        got += [(r, "random") for r in rest[: a.per_lang - len(got)]]
        for r, k in got:
            picked.append(("gen-" + hashlib.sha1((r["id"] + r["lang"]).encode()).hexdigest()[:12], r, {"stratum": k})); counts[(lang, k)] += 1
    n1 = write("stage1_scoring", picked)
    (out / "SYSTEM_PROMPT.txt").write_text(SYSTEM, encoding="utf-8")
    (out / "selection.json").write_text(json.dumps({f"{l}/{k}": n for (l, k), n in sorted(counts.items())}, indent=1))
    print(json.dumps({"stage0_calibration": n0, "stage1_scoring": n1, "languages": len(by)}))


if __name__ == "__main__":
    main()
