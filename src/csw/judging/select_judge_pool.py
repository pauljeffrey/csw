"""Set aside the rows that the judges will score: per language an equal number of good / medium / bad rows (by a rule-based proxy), split into
    eval   (default 100 per language: the gold EVAL set; scored by BOTH judges, with corrections)
    labeled(default 200 per language: scored by the cheap judge only; labels + corrections train the quality estimator)
The remaining generated rows are the unlabeled pool for semi-supervised learning.  NOTHING IS SENT; this only chooses and writes ids.

python -m csw.judging.select_judge_pool --rows DIR --out OUT [--eval-per-lang 100] [--labeled-per-lang 200]

Proxy tiers (the true quality is unknown until a judge reads the rows; the proxy only has to spread the rows over the quality range):
  bad    : a hard gate fails (wrong script, English leak, hm == m_text, pidgin formal English, ...) OR rule_score <= 0.85 OR a pin-violation / over-mixing warning
  good   : no gate failure, no warning, rule_score >= 0.97, min self-confidence >= 0.9, model difficulty <= 3
  medium : everything else
If a language lacks rows of a tier the tier is filled from the nearest tier (lowest-scoring medium rows for bad, highest-scoring medium rows for good) and the shortfall is recorded.
Inside a tier rows are picked round-robin over (pinned switch language, length band) so the eval set covers the generator's scenario space.  One English sentence is used once.
"""
from __future__ import annotations

from csw.paths import DATA_ROOT
import argparse
import collections
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = DATA_ROOT
from csw.datagen import gates                                   # noqa: E402
from csw.judging import score_rows as sr                        # noqa: E402
from csw.datagen.profiles import P, TARGETS, HOLD          # noqa: E402

PIN_WARN = {"english_when_not_pinned", "hm_over_mixed", "hm_repetitive", "pidgin_leak_in_hm", "ood_english_heavy", "m_text_leak"}


def features(r: dict) -> dict | None:
    try:
        s = sr.score_row(r); g = gates.check(dict(r), P[r["lang"]])
    except Exception:
        return None
    conf = r.get("conf") if isinstance(r.get("conf"), dict) else {}
    cmin = min([v for v in conf.values() if isinstance(v, (int, float))] or [0.9])
    diff = r.get("difficulty") if isinstance(r.get("difficulty"), (int, float)) else 3
    warn = set(g["warn"])
    if r["lang"] in ("pcm", "gpe", "wes"):                         # Pidgin is written with English-looking words: these warnings do not mean a defect there
        warn -= {"ood_english_heavy", "m_text_leak", "english_when_not_pinned"}
    return {"rule_score": s["rule_score"], "gate_fail": g["fail"], "gate_warn": sorted(warn), "conf_min": cmin, "difficulty": diff, "has_flags": bool(r.get("flags"))}


def assign_tiers(rows: list[dict]) -> None:
    """Tiers are relative inside the language (absolute thresholds starve languages whose self-confidence is lower): bad = hard failure, pin / mixing warning or the lowest 10% of rule scores;
    good = no warning, no flags, difficulty <= 3 and rule score in the top 30% of the language; medium = the rest."""
    sc = sorted(r["_f"]["rule_score"] for r in rows)
    lo, hi = sc[int(0.10 * (len(sc) - 1))], sc[int(0.70 * (len(sc) - 1))]
    for r in rows:
        f = r["_f"]; warn = set(f["gate_warn"])
        if f["gate_fail"] or warn & PIN_WARN or f["rule_score"] <= lo:
            f["tier"] = "bad"
        elif not warn and not f["has_flags"] and f["difficulty"] <= 3 and f["rule_score"] >= hi:
            f["tier"] = "good"
        else:
            f["tier"] = "medium"


def round_robin(rows: list[dict], n: int, rnd: random.Random) -> list[dict]:
    groups = collections.defaultdict(list)
    for r in rows:
        sc = r.get("scenario") or {}
        groups[(sc.get("switch"), sc.get("length"))].append(r)
    for v in groups.values():
        rnd.shuffle(v)
    keys = sorted(groups, key=lambda k: str(k)); rnd.shuffle(keys)
    out = []
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--eval-per-lang", type=int, default=100); ap.add_argument("--labeled-per-lang", type=int, default=200); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pairs-per-lang", type=int, default=0, help="extra eval pairs per language: a repaired row AND its original (old hm), both judged")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_tier_total = (a.eval_per_lang + a.labeled_per_lang) // 3
    by = collections.defaultdict(lambda: collections.defaultdict(list)); seen = set(); n_in = 0
    for f in sorted(glob.glob(str(Path(a.rows) / "*.raw.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") or not all(r.get(k) for k in ("eng", "m_text", "hm", "ood")) or r["lang"] in HOLD or r["lang"] not in P or r["lang"] not in TARGETS:
                continue
            key = (r["lang"], " ".join(r["eng"].lower().split()), bool(r.get("parent_id")))      # a repaired variant (-hm2) shares its parent's English sentence on purpose
            if key in seen:
                continue
            seen.add(key); n_in += 1
            ft = features(r)
            if ft:
                r["_f"] = ft; by[r["lang"]]["_all"].append(r)
    byid = {}
    for lang in by:
        for r in by[lang]["_all"]:
            byid[r["id"]] = r
        assign_tiers(by[lang]["_all"])
        for r in by[lang]["_all"]:
            by[lang][r["_f"]["tier"]].append(r)
        del by[lang]["_all"]
    rows_out, report = [], {}
    for lang in sorted(by):
        rnd = random.Random(f"{a.seed}-{lang}"); t = by[lang]
        for k in t:
            t[k].sort(key=lambda r: r["_f"]["rule_score"])
        pick = {}
        short = {}
        for tier in ("bad", "good", "medium"):
            pool = list(t.get(tier, [])); got = round_robin(pool, per_tier_total, rnd)
            used = {id(x) for x in got}
            if len(got) < per_tier_total:                                     # fill from the neighbouring tier, lowest / highest scores first
                med = [r for r in t.get("medium", []) if id(r) not in used and id(r) not in {id(x) for v in pick.values() for x in v}]
                med = med if tier == "bad" else med[::-1]
                if tier != "medium":
                    extra = med[: per_tier_total - len(got)]; got += extra; short[tier] = per_tier_total - (len(got) - len(extra))
            pick[tier] = got
        for tier, got in pick.items():
            rnd.shuffle(got)
            n_eval = round(a.eval_per_lang / 3)                               # 33 / 33 / 33 (+1 on one tier so the language has exactly eval-per-lang)
            if tier == "good":
                n_eval = a.eval_per_lang - 2 * round(a.eval_per_lang / 3)
            for i, r in enumerate(got):
                rows_out.append({"id": r["id"], "lang": lang, "tier": tier, "split": "eval" if i < n_eval else "labeled", "proxy": r["_f"], "scenario": r.get("scenario"),
                                 "english_sha1": hashlib.sha1(" ".join(r["eng"].lower().split()).encode()).hexdigest()})
        if a.pairs_per_lang:
            have = {r["id"] for r in rows_out}
            kids = [r for r in byid.values() if r["lang"] == lang and r.get("parent_id") in byid and r["id"] not in have and r["parent_id"] not in have]
            for r in round_robin(kids, a.pairs_per_lang, rnd):
                for x, role in ((byid[r["parent_id"]], "original"), (r, "repaired")):
                    rows_out.append({"id": x["id"], "lang": lang, "tier": x["_f"]["tier"], "split": "eval", "pair": {"role": role, "parent": r["parent_id"], "child": r["id"]}, "proxy": x["_f"], "scenario": x.get("scenario"),
                                     "english_sha1": hashlib.sha1(" ".join(x["eng"].lower().split()).encode()).hexdigest()})
        report[lang] = {"available": {k: len(v) for k, v in t.items()}, "picked": {k: len(v) for k, v in pick.items()}, "short_filled_from_medium": short}
    with open(out / "judge_pool.jsonl", "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "eval_english_sha1.txt").write_text("\n".join(r["english_sha1"] for r in rows_out if r["split"] == "eval") + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    tot = collections.Counter((r["split"], r["tier"]) for r in rows_out)
    print(json.dumps({"rows_considered": n_in, "languages": len(report), "eval": sum(v for (s, _), v in tot.items() if s == "eval"), "labeled": sum(v for (s, _), v in tot.items() if s == "labeled")}))
    short = {l: v["short_filled_from_medium"] for l, v in report.items() if v["short_filled_from_medium"]}
    print("languages short of a tier:", short)


if __name__ == "__main__":
    main()
