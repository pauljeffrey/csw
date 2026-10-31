"""Scoring pass for the batch-audit proposals (the batch route returned corrections but null scores / proficiency).
For each row that has a usable correction and no scores: ask the model to score the ORIGINAL and the CORRECTED row and to report its proficiency in the language.
python -m csw.judging.audit_score [--model google/gemini-3.1-pro-preview] [--conc 6] [--limit N]   ->  exemplars/gold_audit_scores.gemini.jsonl"""
from csw.paths import DATA_ROOT
import json, sys, time, concurrent.futures as cf
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import repair_gold as rg
from csw.datagen import cs_prompts as pr
from csw.datagen import seeds as seedmod
from csw.datagen.profiles import P

SYS = rg.SYSTEM.split("SCORING (1-10")[0] + """SCORING (1-10 per field): 10 = a native expert would publish it unchanged; 8 = good, minor slips; 6 = understandable but with clear errors (wrong words, over/under-mixing, missing diacritics); 4 = mostly wrong or partly in another language; 1-2 = wrong language, copied from another language, or meaningless. eng: formal, neutral, complete. m_text: faithful, grammatical, real target language. hm: pin discipline, 15-40% switch, base language leads, natural switch points. ood: native richness, idiom, correct language.

You are given the ORIGINAL row and a PROPOSED CORRECTION written by an earlier pass. Score both honestly - do not assume the correction is better; check it against the ATTESTED REAL TEXT and your own knowledge, and penalise invented or doubtful words in it. Give a verdict on the correction: "accept" (use it as is), "accept_with_review" (probably right but a native speaker should check the listed points), "keep_original" (the original was better or the correction is not reliable), "unsalvageable" (neither version can be trusted).

LANGUAGE_PROFICIENCY - be honest and conservative, this decides which languages get human reviewers: level = "fluent" / "good" / "limited" / "poor"; can_write_reliably = true/false; weak_areas = short note. Also give switch_language_level for the pinned switch language when it is not English.

Return ONE flat JSON object and nothing else:
{"scores_original": {"eng": 0, "m_text": 0, "hm": 0, "ood": 0}, "scores_correction": {"eng": 0, "m_text": 0, "hm": 0, "ood": 0},
 "verdict": "accept|accept_with_review|keep_original|unsalvageable",
 "level": "fluent|good|limited|poor", "can_write_reliably": true, "weak_areas": "...", "switch_language_level": "fluent|good|limited|poor|n/a",
 "review_points": "words or forms a native speaker must check (or empty)"}"""


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()] if Path(p).exists() else []


def usable(p):
    return p.get("verdict") not in (None, "error") and all(isinstance(p.get(k), str) and p[k] for k in ("eng", "m_text", "hm", "ood"))


def user_msg(r):
    p = P[r["lang"]]; pp = r["proposal"]
    sd = [s["text"] for s in seedmod.load(r["lang"]) if s["track"] == "B" and 20 <= len(s["text"].split()) <= 80][:5]
    ch = pp.get("changes")
    ch = "\n".join(f"- {c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)}" for c in ch) if isinstance(ch, list) else str(ch or "")
    return (f"TARGET LANGUAGE: {p.name} [{p.code}] | PINNED SWITCH LANGUAGE: {r['switch']} | DIALECT: {r.get('dialect') or 'standard'} | GENRE: {r['genre']}\nORTHOGRAPHY: {p.orthography}\n"
            + (f"AUTOMATIC FLAG: {r['flag_reason']}\n" if r.get("flag_reason") else "")
            + "\nORIGINAL ROW:\n" + json.dumps({k: r[k] for k in ("eng", "m_text", "hm", "ood")}, ensure_ascii=False, indent=1)
            + "\n\nPROPOSED CORRECTION:\n" + json.dumps({k: pp[k] for k in ("eng", "m_text", "hm", "ood")}, ensure_ascii=False, indent=1)
            + f"\n\nCHANGES THE EARLIER PASS SAYS IT MADE:\n{ch}\n"
            + (f"\nATTESTED REAL {p.name.upper()} TEXT (authoritative):\n" + "\n".join("- " + s for s in sd) if sd else "") + "\n\nScore and judge. Return the JSON object.")


def main():
    a = sys.argv[1:]; opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    model, conc, limit = opt("--model", "google/gemini-3.1-pro-preview"), int(opt("--conc", 6)), int(opt("--limit", 0))
    src = []
    for f in ("gold_audit_flagged", "gold_audit_rest", "gold_audit_retry"):
        src += read(HERE / "exemplars" / f"{f}.gemini.jsonl")
    best = {}
    for r in src:
        k = (r["lang"], r["eng"], r["m_text"], r["hm"], r["ood"])
        if usable(r["proposal"]) and (k not in best or not isinstance(r["proposal"].get("scores_before"), dict)): best[k] = r
    rows = [r for r in best.values() if not (isinstance(r["proposal"].get("scores_before"), dict) and isinstance(r["proposal"].get("language_proficiency"), dict))]
    out = HERE / "exemplars" / "gold_audit_scores.gemini.jsonl"
    done = {(r["lang"], r["eng"], r["m_text"]) for r in read(out)}
    todo = [r for r in rows if (r["lang"], r["eng"], r["m_text"]) not in done][: limit or None]
    print(len(rows), "rows need scoring;", len(todo), "to do", flush=True)
    fo = out.open("a", encoding="utf-8"); import threading; lk = threading.Lock()
    def one(r):
        d = rg.call(model, SYS, user_msg(r), max_tokens=8000)
        msg = d["choices"][0]["message"].get("content") or ""
        s = pr.parse_json(msg) or {"verdict": "error", "raw": msg[:400]}
        rec = {k: r[k] for k in ("lang", "genre", "switch", "dialect", "eng", "m_text", "hm", "ood") if k in r}; rec["score"] = s; rec["usage"] = d.get("usage")
        with lk: fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
    rg.EFFORT = "low"
    with cf.ThreadPoolExecutor(conc) as ex: list(ex.map(one, todo))


if __name__ == "__main__":
    main()
