"""Enlarge the eng->m replay from the cleared candidate pool (same allow-list of sources as replay2, same exclusions: no NC / FineTranslations / SSA-MTE / decon-flagged), per-language cap, round-robin over sources.
python -m csw.sft.build_replay3 --cap 1500 --out work/replay3.jsonl"""
import argparse, collections, hashlib, json, random, re
from pathlib import Path

EN = set("the and of to is in that it for was with as on be are this but have from or by at an not they we you he she his her their our your what which who when there been has had will would can could should about into than them these those then so if its were also more".split())


def sha(t): return hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()


def ok(e, m):
    we, wm = e.split(), m.split()
    if len(we) < 3 or len(wm) < 2 or len(e) > 600 or len(m) > 900: return False
    r = len(m) / max(1, len(e))
    if r < 0.4 or r > 2.5 or e.strip().lower() == m.strip().lower(): return False
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=1500); ap.add_argument("--out", default="work/replay3.jsonl"); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--mt560-max", type=float, default=0.25, help="max share of mt560 pairs per language (the report flags their licence as unclear and JW300/Bible-heavy); replay2 rows are kept, only new rows are limited"); a = ap.parse_args()
    r2 = [json.loads(l) for l in open("work/replay2.jsonl")]; langs = sorted({r["lang"] for r in r2}); allowed = set(json.load(open("work/replay2_sources.json")))
    have = {(sha(r["eng"]), r["lang"]) for r in r2}; cnt = collections.Counter(r["lang"] for r in r2); rnd = random.Random(a.seed)
    mt = collections.Counter(r["lang"] for r in r2 if "mt560" in r.get("origin", ""))
    pool = collections.defaultdict(lambda: collections.defaultdict(list)); st = collections.Counter()
    for l in open("work/replay2_candidates.jsonl", encoding="utf-8"):
        c = json.loads(l); lang = c["iso"]
        if lang not in langs or c["source_dataset"] not in allowed: continue
        if c["decon_flag"] or c["licence"] in ("non_commercial", "share_alike") or "finetranslations" in c["source_dataset"].lower(): st["excluded"] += 1; continue
        if (sha(c["english"]), lang) in have: st["already_in_replay2"] += 1; continue
        if not ok(c["english"], c["target"]): st["quality_filter"] += 1; continue
        if EN and sum(w in EN for w in re.findall(r"\w+", c["target"].lower())) / max(1, len(c["target"].split())) > 0.25 and lang not in ("pcm", "wes", "gpe", "afr"): st["target_looks_english"] += 1; continue
        pool[lang][c["source_dataset"]].append(c)
    extra = []
    for lang in langs:
        need = a.cap - cnt[lang]; srcs = {k: rnd.sample(v, len(v)) for k, v in pool[lang].items()}; mt_room = max(0, int(a.mt560_max * a.cap) - mt[lang])
        while need > 0 and any(srcs.values()):
            for k in list(srcs):
                if "mt560" in k and mt_room <= 0: srcs[k] = []; continue
                if srcs[k] and need > 0:
                    if "mt560" in k: mt_room -= 1
                    c = srcs[k].pop(); extra.append({"lang": lang, "eng": c["english"], "m": c["target"], "source": "clean_sft", "origin": c["source_dataset"], "licence": c["licence"]}); need -= 1
    seen = set(); out = list(r2)
    for r in extra:
        k = (sha(r["eng"]), r["lang"])
        if k not in seen: seen.add(k); out.append(r)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = collections.Counter(r["lang"] for r in out)
    print(dict(st), "replay2", len(r2), "-> replay3", len(out), "(+%d)" % (len(out) - len(r2))); print("per language:", dict(sorted(per.items())))
    print("origins (new rows):", collections.Counter(r["origin"] for r in extra).most_common(8))


if __name__ == "__main__":
    main()
