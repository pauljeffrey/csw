"""Cheap evaluation of csw generations (no model, no API): chrF++ on the FINAL text only, plus format and rule checks.
    python -m csw.sft.eval_chrf --gens gens_a.jsonl gens_b0.jsonl ... --out results.json [--name sft1]
Per generation the plan is removed (language tags, <draft>..</draft>, <dev>..</dev>) so a format with glosses or a draft cannot move chrF; what is compared is the text after </dev>.
Reported per task and language, macro-averaged over languages (greedy sample 0 and sampled sample 1 separately):
  chrf         corpus chrF++ (sacrebleu, word_order 2) of final text against the reference's final text
  format_ok    share of generations with one <dev>..</dev> block, at least 2 devices, a non-empty final text; with draft on also a <draft>..</draft> block before <dev>
  dev_in_text  mean share of the generation's own devices (glosses stripped) that occur in its final text (hm: the switch phrases; ood: the idioms)
  en_rate      share of word tokens that are common English function words in the final text, minus the same share in the reference ("English leak"), skipped where English is allowed
  len_ratio    generation / reference length in characters
"""
from __future__ import annotations
import argparse, collections, json, re, statistics, unicodedata

TAGS = re.compile(r"^(?:<[a-z]{3}>)+")
DRAFT = re.compile(r"<draft>(.*?)</draft>", re.S)
DEV = re.compile(r"<dev>(.*?)</dev>", re.S)
EN = set("the and of to is in that it for was with as on be are this but have from or by at an not they we you he she his her their our your what which who when there been has had will would can could should about into than them these those then so if its were also more".split())
EN_OK = {"pcm", "gpe", "wes", "afr"}                                        # English is (close to) native here


def parts(t: str):
    t = t.strip(); draft = None
    m = DRAFT.search(t)
    if m: draft = m.group(1).strip(); t = t[:m.start()] + t[m.end():]
    m = DEV.search(t); devs = []
    if m:
        devs = [re.sub(r"\s*\([^)]*\)\s*$", "", x).strip() for x in m.group(1).split(" | ") if x.strip()]
        t = t[:m.start()] + t[m.end():]
    t = TAGS.sub("", t.strip()); t = re.sub(r"<[^>]{1,12}>", " ", t)
    return " ".join(t.split()), devs, draft, bool(m)


def words(t): return re.findall(r"\w+", t.lower())


def en_share(t):
    w = words(t); return sum(x in EN for x in w) / max(1, len(w))


def nfc(t): return unicodedata.normalize("NFC", t).lower()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gens", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", default="")
    ap.add_argument("--draft-on", action="store_true", help="the prompts asked for a draft (arm C test_on)"); a = ap.parse_args()
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2); recs = []
    for f in a.gens:
        for l in open(f, encoding="utf-8"):
            if l.strip(): recs.append(json.loads(l))
    cell = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        ref, _, _, _ = parts(r["reference"]); key = (r["task"], r["lang"])
        for k, g in enumerate(r["generations"][:2]):
            txt, devs, draft, has_dev = parts(g)
            ok = has_dev and len(devs) >= 2 and len(txt) > 0 and (draft is not None or not a.draft_on)
            c = cell[key][k]; c.append({"hyp": txt, "ref": ref, "ok": ok, "dev": (sum(nfc(d) in nfc(txt) for d in devs) / len(devs)) if devs else None,
                                       "en": (en_share(txt) - en_share(ref)) if r["lang"] not in EN_OK and r["task"] != "eng_m" else None, "len": len(txt) / max(1, len(ref))})
    res = {"name": a.name, "n": len(recs), "cells": {}, "macro": {}}
    for (task, lang), ks in cell.items():
        for k, rows in ks.items():
            if task == "eng_m": hyp = [x["hyp"].split(" ", 1)[-1] for x in rows]
            else: hyp = [x["hyp"] for x in rows]
            refs = [x["ref"].split(" ", 1)[-1] if task == "eng_m" else x["ref"] for x in rows]
            m = {"n": len(rows), "chrf": chrf.corpus_score(hyp, [refs]).score, "format_ok": sum(x["ok"] for x in rows) / len(rows),
                 "dev_in_text": statistics.mean([x["dev"] for x in rows if x["dev"] is not None] or [float("nan")]),
                 "en_rate": statistics.mean([x["en"] for x in rows if x["en"] is not None] or [float("nan")]), "len_ratio": statistics.mean(x["len"] for x in rows)}
            res["cells"][f"{task}|{lang}|s{k}"] = m
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, m in res["cells"].items():
        task, lang, s = key.split("|")
        for kk in ("chrf", "format_ok", "dev_in_text", "en_rate", "len_ratio"):
            if m[kk] == m[kk]: by[(task, s)][kk].append(m[kk])
    for (task, s), d in by.items(): res["macro"][f"{task}|{s}"] = {k: round(statistics.mean(v), 4) for k, v in d.items()} | {"languages": len(d["chrf"])}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{'task':9}{'sample':>7}{'langs':>6}{'chrF':>8}{'fmt_ok':>8}{'dev_in':>8}{'en_rate':>9}{'len':>6}")
    for k, m in sorted(res["macro"].items()):
        t, s = k.split("|"); print(f"{t:9}{s:>7}{m['languages']:>6}{m['chrf']:>8.2f}{m['format_ok']:>8.3f}{m.get('dev_in_text', float('nan')):>8.3f}{m.get('en_rate', float('nan')):>9.4f}{m['len_ratio']:>6.2f}")


if __name__ == "__main__":
    main()
