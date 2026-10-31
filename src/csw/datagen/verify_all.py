"""Pre-flight checks before spending GPU time.  python -m csw.datagen.verify_all"""
from csw.paths import DATA_ROOT
import gzip, json, re, sys, collections as C, statistics as st
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import cs_prompts as pr
from csw.datagen import cs_pipeline as pl
from csw.datagen.profiles import P, TARGETS, GOLD_N, HIGH, BRIEF, SENT_RANGE
ok = True
def chk(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
gold = pr.load_gold(); sysm = pr.static_system(gold)
chk("every target language has a profile", set(TARGETS) <= set(P), f"{len(TARGETS)} languages, {sum(TARGETS.values()):,} rows")
chk("no uncorrected gold (status key) is loaded", all("status" not in r for rows in pr.load_aug_gold().values() for r in rows), f"{sum(len(v) for v in pr.load_aug_gold().values())} augmented + {len(gold)} universal gold rows")
chk("system prompt has the formalisation examples, difficulty rule and 40% cap", all(s in sysm for s in ("HOW TO FORMALISE eng", "difficulty: ONE number", "never more than about 40%", "eng_initial")), f"~{len(sysm)//3} tokens")
chk("output shape ends with difficulty", sysm.rstrip().endswith('"flags": [], "difficulty": 0}'))
miss_seed = [l for l in TARGETS if not (HERE / "seeds" / f"{l}.vast.jsonl").exists() and not (HERE / "seeds" / f"{l}.jsonl").exists() and l not in ("gpe", "wes")]
chk("every language has native seeds (except gpe/wes)", not miss_seed, str(miss_seed))
chk("every language has anchors", all((HERE / "anchors" / f"{l}.jsonl").exists() for l in TARGETS))
no_gold = [l for l in TARGETS if GOLD_N.get(l, 0) == 0]
chk("every language has at least one corrected example", not no_gold, str(no_gold))
few = {l: GOLD_N.get(l, 0) for l in HIGH if GOLD_N.get(l, 0) < 3}
print(f"NOTE high-resource languages with fewer than 3 corrected examples: {few}")
tot = 0; maxtok = 0; tr = C.Counter(); dup = 0
for l in TARGETS:
    plan = pl.make_plan(l, TARGETS[l]); tot += len(plan); tr.update(c["track"] for c in plan); dup += len(plan) - len({c["id"] for c in plan})
    if l in BRIEF: assert all(c["length"] == "brief" for c in plan)
    assert all(SENT_RANGE[c["length"]][0] <= c["n_sent"] <= SENT_RANGE[c["length"]][1] for c in plan), l
chk("plans: unique ids everywhere", dup == 0, f"{tot:,} rows; tracks {dict(tr)}")
sizes = []
for f in sorted((HERE / "prompts_out").glob("*.jsonl.gz")) if (HERE / "prompts_out").exists() else []:
    for l in gzip.open(f, "rt", encoding="utf-8"):
        r = json.loads(l); sizes.append((len(r["user"]) // 3, r["max_tokens"], f.name.split(".")[0], r["system_tokens_est"]))
if sizes:
    worst = max(sizes, key=lambda x: x[3] + x[0] + x[1])
    chk("context budget: system + card + max_tokens <= 32768", worst[3] + worst[0] + worst[1] <= 32768, f"worst case {worst[3] + worst[0] + worst[1]} tokens ({worst[2]}); card tokens p50 {st.median(s[0] for s in sizes):.0f}, p99 {sorted(s[0] for s in sizes)[int(.99*len(sizes))]}, max {max(s[0] for s in sizes)}")
    chk("exported prompts cover every planned row", len(sizes) == tot, f"{len(sizes):,} exported / {tot:,} planned")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
