"""Round 2 of judging: more labelled rows, each row judged by ONE judge, only in the languages that judge handles, no row (and no English sentence) judged twice.
python -m csw.judging.select_round2 --rows work/post_rows --done work/judge_pool_v2/judge_pool.jsonl --out work/judge_pool_r2 --flash-rows 22000 --luna-rows 19500
Per language the row count is the judge's budget spread equally over its languages; where a language has too few unjudged sentences the two judges split what exists and the leftover budget goes to the other languages.
Rows are drawn from the proxy tiers (bad / medium / good) in equal thirds, round-robin over (pin, length) as in select_judge_pool."""
from csw.paths import DATA_ROOT
import argparse, collections, glob, hashlib, json, random, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import select_judge_pool as S
from csw.datagen.profiles import P, TARGETS, HOLD

FLASH_SKIP = {"bam", "ewe"}                                           # Flash flags many rows it cannot fix (nn 8-9%), low grades, weak repair signal
LUNA_SKIP = {"bam", "ewe", "fon", "ful", "kik", "lug", "luo", "nya", "run", "tso", "wol", "tum"}   # capability screen + tum (37% of pool rows self-rated limited)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--done", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--flash-rows", type=int, default=22000); ap.add_argument("--luna-rows", type=int, default=19500); ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
    done = [json.loads(l) for l in open(a.done)]
    done_ids = {r["id"] for r in done}; done_eng = {r["english_sha1"] for r in done}
    by = collections.defaultdict(list); seen = set()
    for f in sorted(glob.glob(str(Path(a.rows) / "*.raw.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("error") or not all(r.get(k) for k in ("eng", "m_text", "hm", "ood")) or r["lang"] in HOLD or r["lang"] not in P or r["lang"] not in TARGETS: continue
            sha = hashlib.sha1(" ".join(r["eng"].lower().split()).encode()).hexdigest()
            if r["id"] in done_ids or sha in done_eng or (r["lang"], sha) in seen or r.get("parent_id"): continue      # repaired -hm2 rows were judged as pairs; keep their originals out of round 2 too
            seen.add((r["lang"], sha)); ft = S.features(r)
            if ft: r["_f"] = ft; r["_sha"] = sha; by[r["lang"]].append(r)
    langs = sorted(by); avail = {l: len(by[l]) for l in langs}
    fl = [l for l in langs if l not in FLASH_SKIP]; lu = [l for l in langs if l not in LUNA_SKIP]
    qf = {l: 0 for l in langs}; ql = {l: 0 for l in langs}; cap = dict(avail)
    for _ in range(6):                                                 # water-filling: equal quota, capped by what exists, leftover redistributed
        for q, ls, tot in ((qf, fl, a.flash_rows), (ql, lu, a.luna_rows)):
            left = tot - sum(q.values()); free = [l for l in ls if q[l] + ql[l] + qf[l] - q[l] < cap[l]]
            if not free or left <= 0: continue
            for l in free: q[l] += min(left // len(free), max(0, cap[l] - qf[l] - ql[l]))
    rows_out = []; rep = {}
    for l in langs:
        rows = by[l]; S.assign_tiers(rows); rnd = random.Random(f"{a.seed}-{l}"); tiers = collections.defaultdict(list)
        for r in rows: tiers[r["_f"]["tier"]].append(r)
        want = {"flash": qf[l], "luna": ql[l]}; pick = {"flash": [], "luna": []}; used = set()
        for judge in ("luna", "flash"):                                  # luna first: it has the smaller language set
            n = want[judge]
            for t in ("bad", "good", "medium"):
                pool = [r for r in tiers[t] if r["id"] not in used]; got = S.round_robin(pool, n // 3 + (1 if t == "bad" and n % 3 else 0), rnd)
                for r in got: used.add(r["id"]); pick[judge].append(r)
            short = n - len(pick[judge])
            if short > 0:
                rest = [r for r in rows if r["id"] not in used]; rnd.shuffle(rest)
                for r in rest[:short]: used.add(r["id"]); pick[judge].append(r)
        for judge, got in pick.items():
            for r in got:
                rows_out.append({"id": r["id"], "lang": l, "tier": r["_f"]["tier"], "split": "r2", "judge": judge, "proxy": r["_f"], "scenario": r.get("scenario"), "english_sha1": r["_sha"]})
        rep[l] = {"available": avail[l], "flash": len(pick["flash"]), "luna": len(pick["luna"])}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "judge_pool.jsonl", "w", encoding="utf-8") as f:
        for r in rows_out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print("flash", sum(v["flash"] for v in rep.values()), "luna", sum(v["luna"] for v in rep.values())); print({l: (v["available"], v["flash"], v["luna"]) for l, v in rep.items()})


if __name__ == "__main__":
    main()
