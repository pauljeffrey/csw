"""What the two LLM judges said about a row, turned into (a) corrected copies of the row, (b) the fields they rejected, (c) a curated evaluation set.

    from csw.sft import judge_fixes as J
    runs = J.load()                                   # {row_id: {"row": row, "judges": {judge: reply}}}
    variants, bad = J.variants_and_bad(runs)          # corrected row copies; {row_id: {"m_text","hm","ood"} fields a judge called major/critical}
    ids = J.eval_ids(runs)                            # rows both judges (or the one that reads the language) found error-free and top-graded

A judge fix is trusted only when the judge read the language (r=y), is fluent/good (l f|g), is not "cannot correct" (nn 0) and is at least half sure (k >= .5).
Fixes are the judge's `fx` (whole field) or `ed` (exact-once word replacements); both must keep the digits, stay within 0.5-2x the original length and be non-empty,
otherwise that field stays uncorrected (and, when the judge called it major/critical, the original field is withheld from SFT)."""
from __future__ import annotations
from csw.paths import DATA_ROOT
import json, re, unicodedata
from pathlib import Path

HERE = DATA_ROOT
FIELD = {"m": "m_text", "h": "hm", "o": "ood"}
RUNS = [("flash", "work/pool_flash", "gemini-3.8-flash"), ("luna", "work/pool_luna", "gpt-5.6-luna"),
        ("flash", "work/pool_r2_flash", "gemini-3.8-flash"), ("luna", "work/pool_r2_luna", "gpt-5.6-luna"),
        ("luna", "work/pool_test_luna", "gpt-5.6-luna")]          # second opinion on the curated test rows (single-judged by Flash before)
NFC = lambda t: unicodedata.normalize("NFC", t)


def load(root: Path = HERE) -> dict:
    """one record per judged row id; the first reply of each judge wins (round 1 had 7 duplicate ids)"""
    out = {}
    for judge, d, model in RUNS:
        res, req = root / d / f"results_{model}@v13.jsonl", root / d / "requests.jsonl"
        if not res.exists() or not req.exists(): continue
        rows = {}
        for l in open(req, encoding="utf-8"):
            if l.strip():
                q = json.loads(l); rows[q["rid"]] = q["row"]
        for l in open(res, encoding="utf-8"):
            if not l.strip(): continue
            x = json.loads(l); rep, row = x.get("reply"), rows.get(x["rid"])
            if not rep or row is None: continue
            rec = out.setdefault(row["id"], {"row": row, "judges": {}})
            rec["judges"].setdefault(judge, rep)
    return out


def trusted(rep: dict) -> bool:
    return rep.get("r") == "y" and rep.get("l") in ("f", "g") and not rep.get("nn") and (rep.get("k") or 0) >= 0.5


def flagged(rep: dict) -> set[str]:
    """fields (m_text|hm|ood) with at least one major or critical error span"""
    return {FIELD[e[0]] for e in rep.get("e") or [] if len(e) >= 4 and e[0] in FIELD and e[3] in ("j", "c")}


def _sane(old: str, new: str) -> bool:
    if not new or not new.strip(): return False
    if sorted(re.findall(r"\d+", old)) != sorted(re.findall(r"\d+", new)): return False
    return 0.5 <= len(new) / max(1, len(old)) <= 2.0


def corrected_fields(row: dict, rep: dict) -> dict:
    """{m_text|hm|ood: corrected text} for the fields this judge's fix applies to cleanly"""
    out = {}
    fx = rep.get("fx") or {}
    for k, name in FIELD.items():
        new = fx.get(k)
        if isinstance(new, str) and NFC(new) != NFC(row[name]) and _sane(row[name], new): out[name] = NFC(new)
    by = {}
    for e in rep.get("ed") or []:
        if len(e) == 3 and e[0] in FIELD and name_ok(e): by.setdefault(FIELD[e[0]], []).append((e[1], e[2]))
    for name, eds in by.items():
        if name in out: continue
        t = NFC(row[name]); spans = []
        for old, new in eds:
            old, new = NFC(old), NFC(new)
            if not old or t.count(old) != 1: spans = None; break
            i = t.index(old); spans.append((i, i + len(old), new))
        if not spans: continue
        spans.sort()
        if any(b[0] < a[1] for a, b in zip(spans, spans[1:])): continue
        new_t = t
        for a0, a1, rep_t in reversed(spans): new_t = new_t[:a0] + rep_t + new_t[a1:]
        if new_t != t and _sane(t, new_t): out[name] = new_t
    return out


def name_ok(e) -> bool:
    return isinstance(e[1], str) and isinstance(e[2], str)


def variants_and_bad(runs: dict) -> tuple[list[dict], dict]:
    variants, bad = [], {}
    for rid, rec in runs.items():
        row = rec["row"]; flag = set()
        for judge, rep in rec["judges"].items():
            if not trusted(rep): continue
            fl = flagged(rep); flag |= fl
            fix = corrected_fields(row, rep)
            if fix:
                v = dict(row, source="generated", id=f"{row['id']}-fx{judge[0]}", parent_id=row["id"], quality_tags=list(row.get("quality_tags") or []) + [f"judge_fix_{judge}"], **fix)
                v["bad_fields"] = sorted(fl - set(fix)); variants.append(v)
        if flag: bad[rid] = flag
    return variants, bad


def grades(rep: dict) -> list[int]:
    return [g for grp in rep.get("g") or [] for g in (grp or []) if isinstance(g, int)]


def eval_score(rec: dict, relax_minor: bool = False):
    """None if the row does not qualify; else (n_judges, mean grade over the trusted judges). Strict: no error span of any severity, no fix, every grade >= 4. Relaxed: minor spans and cosmetic fixes allowed, no major/critical span, every grade >= 4."""
    reps = [r for r in rec["judges"].values() if trusted(r)]
    if not reps: return None
    means = []
    for r in reps:
        errs = [e for e in r.get("e") or [] if len(e) >= 4]
        if (r.get("fx") or r.get("ed")) and not relax_minor: return None     # relaxed rule : minor errors and cosmetic fixes are allowed, major/critical never
        if any(e[3] in ("j", "c") for e in errs): return None
        if errs and not relax_minor: return None
        g = grades(r)
        if not g or min(g) < 4: return None
        means.append(sum(g) / len(g))
    return len(reps), sum(means) / len(means)


def eval_ids(runs: dict, per_lang: int = 120, relax_minor: bool = False) -> dict:
    """{lang: [row ids]} best first: rows judged by both judges before one-judge rows, then by mean grade"""
    by = {}
    for rid, rec in runs.items():
        s = eval_score(rec, relax_minor)
        if s: by.setdefault(rec["row"]["lang"], []).append((s, rid))
    return {lang: [rid for _, rid in sorted(v, reverse=True)[:per_lang]] for lang, v in by.items()}
