"""Dedicated compact judge prompts, one per task, so every request carries only what that task needs (cost).  Strict-reviewer calibration is built in.
  M    translation pairs (english -> language): M1..M6, 8 pairs per request                                  (label_replay.py builds these; text kept here for reference)
  HM   code-switched text only: H1..H9, D1, plan items                                                        (corruption chains of the hm field, no spans)
  OOD  informal idiomatic text only: O1..O7, D2, idioms                                                       (corruption chains of the ood field, no spans)
  BOTH hm + ood of one row: H, O, D1, D2, X1, plan items, idioms, error spans and edits                       (the new 10k rows)
Replies are converted to the 27-slot grade vector used by the evaluator (M slots stay empty for HM / OOD / BOTH)."""
from csw.paths import DATA_ROOT
import json, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.sft import devices as dv
from csw.judging import judge_v11_test as J

SCALE = ("SCALE (strict native reviewer): 5 no flaw a native speaker would notice; 4 exactly one minor flaw; 3 several minor flaws or one clear flaw; 2 a major flaw (meaning, register or language wrong); 1 unusable. "
         "Do not default to 4 or 5: in a typical batch about half of the grades are 3 or lower. Grade only what you can verify, else null. A missing, wrong or extra tone mark or special letter is a spelling error (that dimension at most 3).")
HEAD = "You grade ONE text from an automatic generator for an African-language dataset. Reply with ONE compact JSON object, no prose. "
HM_DIMS = ("H1 fidelity to eng; H2 matrix-language integrity (the base language gives order, agreement, affixes); H3 switch-point grammaticality; H4 pin purity (a third language is an error); H5 density/distribution of switches; "
           "H6 morphological integration of switched items (null if little affixation); H7 fluency/orthography; H8 authenticity (motivated switches, not quota substitution); H9 switch-unit fit (noun/phrase units good; lone function words, bare verbs poor); D1 plan quality (meaningful, minimal, apt units, used).")
OOD_DIMS = ("O1 facts preserved (imagery allowed, no new facts); O2 idiom authenticity and use; O3 colloquial naturalness/imagery; O4 dialect/cultural fit (null unless you can name the cue); O5 language purity (English or third-language leakage is an error); "
            "O6 fluency/orthography; O7 richer than a literal paraphrase, not stuffed; D2 idiom list quality.")
PD = "pd: one value per plan item P1..: 2 apt, correct and really used in h; 1 weak; 0 wrong, misused or absent; null cannot verify."
IQ = "iq: one value per idiom I1..: 2 real, apt, correctly used and glossed; 1 weak; 0 wrong or misused; null cannot verify (never call an idiom wrong only because you do not know it)."
SPANS = ("e: error spans [[field h|o, exact words from the text, code, severity n|j|c, dimension]], worst first, at most 6 in total (codes: MT OM AD UT TN GR SP LL RG TL SG MB PV IN GS SU PL DV FI IM LI CU XI CP). "
         "ed: edits [[field, exact words to replace, replacement]] for every major or critical span; the words must occur exactly once in that field (add neighbouring words until unique), edits must not overlap; change only what is wrong, keep voice, dialect, pin, density and tone marks, never invent words; more than 8 changes -> leave ed empty. "
         "nn: 1 only if you cannot read the language or cannot fix an error without inventing words, else 0.")
SYS_HM = (HEAD + "INPUT: language, pin (the only language allowed besides the base), eng (formal English), h (base-language text with the pin mixed in, 15-40% of words), P1.. (planned switch phrases), rule measurements. "
          "KEYS: r can you read it y/p/n (n: reply {\"r\":\"n\"} only); g [H1..H9,D1] integers 1-5 or null; " + PD + "\n" + HM_DIMS + "\n" + SCALE + "\nSkeleton: {\"r\":\"y\",\"g\":[5,4,4,3,5,null,4,4,4,4],\"pd\":[2,1]}")
SYS_OOD = (HEAD + "INPUT: language, dialect, eng (formal English), o (informal native rendering with idioms/proverbs, same facts), I1.. (idioms with gloss), rule measurements. "
           "KEYS: r can you read it y/p/n (n: reply {\"r\":\"n\"} only); g [O1..O7,D2] integers 1-5 or null; " + IQ + "\n" + OOD_DIMS + "\n" + SCALE + "\nSkeleton: {\"r\":\"y\",\"g\":[5,4,4,null,5,4,4,4],\"iq\":[2]}")
SYS_BOTH = (HEAD + "INPUT: language, dialect, pin, eng (formal English), h (base language with the pin mixed in), o (informal native rendering with idioms), P1.. (planned switch phrases), I1.. (idioms with gloss), rule measurements. "
            "KEYS: r can you read it y/p/n (n: reply {\"r\":\"n\",\"nn\":1} only); g [[H1..H9],[O1..O7],[D1,D2],[X1]] integers 1-5 or null; " + PD + " " + IQ + " " + SPANS + "\nX1 same facts in h and o.\n" + HM_DIMS + "\n" + OOD_DIMS + "\n" + SCALE +
            "\nSkeleton: {\"r\":\"y\",\"g\":[[5,4,4,3,5,null,4,4,4],[5,null,4,null,5,4,4],[4,null],[5]],\"pd\":[2,1],\"iq\":[2],\"e\":[[\"h\",\"word\",\"PV\",\"j\",\"H4\"]],\"ed\":[[\"o\",\"old words\",\"new words\"]],\"nn\":0}")


def _ctx(r, lex):
    p = J.P[r["lang"]]; sc = r.get("scenario") or {}; h = dv.hm_devices(r, lex); plan = h["devices"]; idioms = [d for d in (r.get("devices") or []) if isinstance(d, str)][:8]
    ew = set(J.gates.words(r["eng"])); cs = lambda t: round(sum(x in ew and len(x) >= 4 for x in J.gates.words(t)) / max(1, len(J.gates.words(t))), 2)
    dg = J.digits(r["eng"]); miss = lambda k: [d for d in dg if d not in r[k]]
    return p, sc, h, plan, idioms, cs, miss


def user_hm(r, lex):
    p, sc, h, plan, idioms, cs, miss = _ctx(r, lex); m = {"language": p.name, "pin": sc.get("switch"), "eng": r["eng"], "h": r["hm"], "P": {f"P{i + 1}": x for i, x in enumerate(plan)}, "rules": {"hm_switch_share": h["switch_ratio"], "eng_word_share": cs(r["hm"])}}
    if miss("hm"): m["rules"]["digits_missing"] = miss("hm")
    return json.dumps(m, ensure_ascii=False)


def user_ood(r, lex):
    p, sc, h, plan, idioms, cs, miss = _ctx(r, lex); m = {"language": p.name, "dialect": sc.get("dialect") or "standard", "eng": r["eng"], "o": r["ood"], "I": {f"I{i + 1}": x for i, x in enumerate(idioms)}, "rules": {"eng_word_share": cs(r["ood"])}}
    if miss("ood"): m["rules"]["digits_missing"] = miss("ood")
    return json.dumps(m, ensure_ascii=False)


def user_both(r, lex):
    p, sc, h, plan, idioms, cs, miss = _ctx(r, lex)
    m = {"language": p.name, "dialect": sc.get("dialect") or "standard", "pin": sc.get("switch"), "eng": r["eng"], "h": r["hm"], "o": r["ood"], "P": {f"P{i + 1}": x for i, x in enumerate(plan)}, "I": {f"I{i + 1}": x for i, x in enumerate(idioms)},
         "rules": {"hm_switch_share": h["switch_ratio"], "eng_word_share": {"h": cs(r["hm"]), "o": cs(r["ood"])}}}
    dm = {k: miss(k) for k in ("hm", "ood") if miss(k)}
    if dm: m["rules"]["digits_missing"] = dm
    return json.dumps(m, ensure_ascii=False)


def to_vec(kind, rep):
    """compact reply -> 27-slot vector (None where not graded); None if unusable"""
    if not isinstance(rep, dict) or rep.get("r") == "n": return None
    ok = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and 1 <= x <= 5
    v = [None] * 27; g = rep.get("g")
    if kind == "hm" and isinstance(g, list) and len(g) == 10:
        for i in range(9): v[6 + i] = float(g[i]) if ok(g[i]) else None
        v[22] = float(g[9]) if ok(g[9]) else None
    elif kind == "ood" and isinstance(g, list) and len(g) == 8:
        for i in range(7): v[15 + i] = float(g[i]) if ok(g[i]) else None
        v[23] = float(g[7]) if ok(g[7]) else None
    elif kind == "both" and isinstance(g, list) and len(g) == 4 and all(isinstance(a, list) for a in g):
        for i in range(9): v[6 + i] = float(g[0][i]) if len(g[0]) > i and ok(g[0][i]) else None
        for i in range(7): v[15 + i] = float(g[1][i]) if len(g[1]) > i and ok(g[1][i]) else None
        v[22] = float(g[2][0]) if len(g[2]) > 0 and ok(g[2][0]) else None; v[23] = float(g[2][1]) if len(g[2]) > 1 and ok(g[2][1]) else None; v[24] = float(g[3][0]) if len(g[3]) > 0 and ok(g[3][0]) else None
    else: return None
    for k, i in (("pd", 25), ("iq", 26)):
        a = [x for x in (rep.get(k) or []) if isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 2]
        if a and ((k == "pd" and kind in ("hm", "both")) or (k == "iq" and kind in ("ood", "both"))): v[i] = 1 + 2 * sum(a) / len(a)
    return v
