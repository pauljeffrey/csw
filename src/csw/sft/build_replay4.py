"""Uniform per-language enlargement of the eng->m replay from a larger candidate pool (build_replay2 --cap 3500).
python -m csw.sft.build_replay4 --cand work/replay4_candidates.jsonl --keep work/replay2.jsonl --target 3000 --out work/replay4.jsonl
Per language: keep the replay2 rows, then fill up to --target by round-robin over the non-mt560 sources first; mt560 (licence unclear, JW300/Bible-heavy) only up to --mt560-max of the target,
and beyond that only for languages with no other cleared source (--mt560-fallback) -- the share per language is reported.  Excluded: FineTranslations (model-written English), non-commercial, share-alike (SSA-MTE, FLORES dev halves),
decon-flagged rows, pairs failing a length-ratio / English-leak check.  Held-out English (val/test) is guarded later by csw.sft.add_replay."""
import argparse, collections, hashlib, json, random, re
EN = set("the and of to is in that it for was with as on be are this but have from or by at an not they we you he she his her their our your what which who when there been has had will would can could should about into than them these those then so if its were also more".split())


def sha(t): return hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()


def ok(e, m):
    we, wm = e.split(), m.split()
    if len(we) < 3 or len(wm) < 2 or len(e) > 600 or len(m) > 900: return False
    r = len(m) / max(1, len(e))
    return 0.4 <= r <= 2.5 and e.strip().lower() != m.strip().lower()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cand", required=True); ap.add_argument("--keep", required=True); ap.add_argument("--target", type=int, default=3000); ap.add_argument("--out", required=True)
    ap.add_argument("--mt560-max", type=float, default=0.25); ap.add_argument("--no-mt560-fallback", action="store_true"); ap.add_argument("--seed", type=int, default=21); a = ap.parse_args()
    keep = [json.loads(l) for l in open(a.keep, encoding="utf-8")]; langs = sorted({r["lang"] for r in keep}); have = {(sha(r["eng"]), r["lang"]) for r in keep}
    cnt = collections.Counter(r["lang"] for r in keep); mt = collections.Counter(r["lang"] for r in keep if "mt560" in r.get("origin", "")); rnd = random.Random(a.seed)
    pool = collections.defaultdict(lambda: collections.defaultdict(list)); st = collections.Counter()
    for l in open(a.cand, encoding="utf-8"):
        c = json.loads(l); lang = c["iso"]
        if lang not in langs: continue
        s = c["source_dataset"]
        if c["decon_flag"] or c["licence"] in ("non_commercial", "share_alike") or "finetranslations" in s.lower() or "SSA-MTE" in s: st["excluded_licence_or_flag"] += 1; continue
        if (sha(c["english"]), lang) in have: st["already_in_replay2"] += 1; continue
        if not ok(c["english"], c["target"]): st["quality_filter"] += 1; continue
        if lang not in ("pcm", "wes", "gpe", "afr") and sum(w in EN for w in re.findall(r"\w+", c["target"].lower())) / max(1, len(c["target"].split())) > 0.25: st["target_looks_english"] += 1; continue
        pool[lang][s].append(c)
    new = []; seen = set(have); report = {}
    for lang in langs:
        need = a.target - cnt[lang]; room = max(0, int(a.mt560_max * a.target) - mt[lang])
        other = {k: rnd.sample(v, len(v)) for k, v in pool[lang].items() if "mt560" not in k}; m560 = {k: rnd.sample(v, len(v)) for k, v in pool[lang].items() if "mt560" in k}
        def draw(srcs, n):
            got = []
            while n > 0 and any(srcs.values()):
                for k in list(srcs):
                    while srcs[k]:
                        c = srcs[k].pop(); key = (sha(c["english"]), lang)
                        if key in seen: continue
                        seen.add(key); got.append({"lang": lang, "eng": c["english"], "m": c["target"], "source": "clean_sft", "origin": c["source_dataset"], "licence": c["licence"]}); n -= 1; break
                    if n <= 0: break
            return got
        g1 = draw(other, need); need -= len(g1); g2 = draw(m560, min(need, room)); need -= len(g2); g3 = []
        if need > 0 and not a.no_mt560_fallback: g3 = draw(m560, need); need -= len(g3)         # no other cleared source left: mt560 fills the gap
        new += g1 + g2 + g3; tot = cnt[lang] + len(g1) + len(g2) + len(g3); mtn = mt[lang] + len(g2) + len(g3)
        report[lang] = {"total": tot, "mt560_share": round(mtn / max(1, tot), 2), "short_by": max(0, need)}
    with open(a.out, "w", encoding="utf-8") as f:
        for r in keep + new: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(st), "replay2", len(keep), "->", len(keep) + len(new), "(+%d)" % len(new))
    print("per language (total, mt560 share, short_by):", {k: (v["total"], v["mt560_share"], v["short_by"]) for k, v in report.items()})


if __name__ == "__main__":
    main()
