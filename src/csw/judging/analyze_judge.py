"""Read judge results (run_judge_openrouter output) and report.

python -m csw.judging.analyze_judge --dir judge_batch_v0 --judge gemini-3.8-flash

Stage 0: per corruption kind the detection rate (a field scored <= 6, or meaning_ok false for number / negation, or can_read false counts as 'cannot judge', reported separately)
         and the false-alarm rate on the clean gold copies; per language the same for the languages with enough rows.
Stage 1: per language mean scores and can_read / language_level mix, by stratum (clean / flagged / random), and the share of rows with a problem.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path


def load(path: Path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("parsed") and isinstance(r["parsed"].get("scores"), dict):
                out[r["id"]] = r["parsed"]
    return out


def mn(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return round(st.mean(v), 2) if v else None


def caught(p: dict, kind: str) -> bool:
    s = p["scores"]; low = min([s.get(k, 10) for k in ("m_text", "hm", "ood") if isinstance(s.get(k), (int, float))] or [10])
    mo = p.get("meaning_ok") or {}
    if kind in ("number", "negation"):
        return low <= 5 or any(v is False for v in mo.values())
    return low <= 6 or any(v is False for v in mo.values())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--judge", required=True)
    a = ap.parse_args(); d = Path(a.dir)
    f0 = d / "results" / f"stage0_{a.judge}.jsonl"
    if f0.exists():
        meta = json.loads((d / "stage0_calibration" / "meta.json").read_text()); res = load(f0)
        by = collections.defaultdict(list); bylang = collections.defaultdict(lambda: collections.defaultdict(list)); unread = collections.Counter()
        for cid, p in res.items():
            k = meta[cid]["corruption"]; lang = meta[cid]["lang"]; c = caught(p, k)
            by[k].append(c); bylang[lang][k].append(c); unread[k] += p.get("can_read") is False
        print(f"== Stage 0 ({len(res)}/{len(meta)} parsed) judge={a.judge}")
        print(f"{'kind':10}{'n':>4}{'caught/flagged':>16}{'cannot read':>13}")
        for k in ("clean", "swap", "truncate", "wronglang", "number", "negation"):
            v = by.get(k, [])
            if v:
                print(f"{k:10}{len(v):>4}{100*sum(v)/len(v):>14.0f}% {unread[k]:>11}")
        print("per language: false-alarm rate on clean | catch rate on real corruptions (swap / truncate / wronglang / number / negation)")
        for lang in sorted(bylang):
            c = bylang[lang]; bad = [x for k, v in c.items() if k != "clean" for x in v]
            if len(bad) >= 4:
                fa = c.get("clean", [])
                print(f"  {lang:4} clean n={len(fa):>2} false-alarm {100*sum(fa)/max(1,len(fa)):>3.0f}% | corrupted n={len(bad):>3} caught {100*sum(bad)/len(bad):>3.0f}%")
    f1 = d / "results" / f"stage1_{a.judge}.jsonl"
    if f1.exists():
        meta = json.loads((d / "stage1_scoring" / "meta.json").read_text()); res = load(f1)
        agg = collections.defaultdict(lambda: collections.defaultdict(list)); lvl = collections.defaultdict(collections.Counter)
        for cid, p in res.items():
            lang, s = meta[cid]["lang"], p["scores"]; strat = meta[cid].get("stratum")
            for k in ("m_text", "hm", "ood"):
                agg[lang][k].append(s.get(k)); agg[lang][f"{strat}:{k}"].append(s.get(k))
            lvl[lang][p.get("language_level")] += 1; agg[lang]["unread"].append(p.get("can_read") is False); agg[lang]["problem"].append(bool(p.get("problems")))
        print(f"\n== Stage 1 ({len(res)}/{len(meta)} parsed) judge={a.judge}")
        print("lang   n  m_text  hm   ood | clean-stratum  flagged-stratum (mean of 3 fields) | cannot-read | levels")
        for lang in sorted(agg):
            g = agg[lang]
            cl = mn([x for k in ("m_text", "hm", "ood") for x in g.get(f"clean:{k}", [])]); fl = mn([x for k in ("m_text", "hm", "ood") for x in g.get(f"flagged:{k}", [])])
            print(f"{lang:4}{len(g['m_text']):>5} {mn(g['m_text']):>6} {mn(g['hm']):>5} {mn(g['ood']):>5} | {cl!s:>8} {fl!s:>10} | {100*sum(g['unread'])/len(g['unread']):>4.0f}% | {dict(lvl[lang])}")


if __name__ == "__main__":
    main()
