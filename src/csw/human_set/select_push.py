"""Build the human-preference test set from the scored pool and the test-set prompts, and (with --push) upload everything.
Priority: every test-set item (all 6,230, with the reference, SFT, RL-best and RL-last outputs the Flash judge already graded, plus fresh candidates).
Then water-filling over languages with pool items until TARGET items, stratified by task and by evaluator score.
python select_push.py [--push] [--target 8000]"""
import json, gzip, re, random, unicodedata, collections, statistics, sys, os, importlib.util
TARGET = int(sys.argv[sys.argv.index("--target") + 1]) if "--target" in sys.argv else 8000
W = "/workspace/csw_hs"; OUT = f"{W}/push"; os.makedirs(f"{OUT}/data", exist_ok=True); os.makedirs(f"{OUT}/test_set", exist_ok=True)
sp = importlib.util.spec_from_file_location("ec", f"{W}/eval_chrf.py"); ec = importlib.util.module_from_spec(sp); sp.loader.exec_module(ec)
rd = lambda p: [json.loads(l) for l in (gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8"))]
rng = random.Random(20261101)
T6 = {"m_text": "eng_m", "hm": "eng_hm", "ood": "eng_ood"}
def norm(t): return re.sub(r"[^\w ]", "", unicodedata.normalize("NFC", t).lower()).split()
def grams(t, n=5): s = " ".join(norm(t)); return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}
def jacc(a, b): return len(a & b) / max(1, len(a | b))
mean = lambda x: sum(x) / len(x)

# ---- pool (scored by score_all.py; this is also what gets re-pushed as data/*.jsonl.gz)
pool = []
for t in ("m_text", "hm", "ood"):
    rows = rd(f"{W}/pool/{t}.jsonl.gz")
    with gzip.open(f"{OUT}/data/{t}.jsonl.gz", "wt", encoding="utf-8") as fo:
        for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        for c in r["candidates"]: c["systems"] = ["gen"]
        pool.append({"id": r["id"], "origin": "pool", "task": T6[t], "lang": r["lang"], "switch": r.get("switch"), "english": r["english"], "prompt": r["prompt"], "candidates": r["candidates"]})
for r in rd(f"{W}/pool/eval_m_clean.jsonl.gz"):                                 # translations (rl2c) of the test set's English sources; selected like pool rows, no reference and no Flash grades
    for c in r["candidates"]: c["systems"] = ["gen"]
    pool.append({"id": r["id"], "origin": "test_set_eng_m", "task": "eng_m", "lang": r["lang"], "switch": None, "english": r["english"], "prompt": r["prompt"], "candidates": r["candidates"]})
print("pool rows", len(pool))

# ---- test-set items with all their candidates
test = {}
for l in open(f"{W}/data/test_off.jsonl"):
    r = json.loads(l); test[r["id"]] = r
flash = json.load(gzip.open(f"{W}/data/scores.json.gz", "rt")); evs = {}
for r in rd(f"{W}/data/evaluator_rules_per_item.jsonl.gz"): evs[f"{r['id']}|{r['system']}"] = r["evaluator"]
def sysout(path, name):
    d = {}
    for r in rd(path):
        text, devs, _, has = ec.parts(r["generations"][0]); d[r["id"]] = {"text": text, "devices": devs, "tags": re.findall(r"<([a-z]{3,5})>", r["generations"][0].split("</dev>")[0]), "raw": r["generations"][0], "ok": has, "systems": [name]}
    return d
SYS = {"sft": sysout(f"{W}/data/full_E_test_off.jsonl", "sft"), "rl_last": sysout(f"{W}/data/rl7last_test_off.jsonl", "rl_last")}
ref = {r["id"]: r["candidates"][0] for r in rd(f"{W}/pool/eval_ref.jsonl.gz")}
new = {r["id"]: r for r in rd(f"{W}/pool/eval_clean.jsonl.gz")}
evalrows = []; dupes = 0
for i, r in test.items():
    if i not in ref: continue
    cands = []
    def add(c, systems, ev, greedy=False, k=None):
        global dupes
        g = grams(c["text"])
        for o in cands:
            if jacc(g, o["_g"]) >= 0.9: o["systems"] = sorted(set(o["systems"] + systems)); dupes += 1; o["greedy"] = o["greedy"] or greedy; return
        cands.append({"text": c["text"], "devices": c["devices"], "tags": c.get("tags", []), "systems": systems, "greedy": greedy, "ev": ev, "_g": g})
    c = ref[i]; add(c, ["reference"], c.get("ev"))
    for s in ("sft", "rl_last"):
        c = SYS[s].get(i)
        if c and c["ok"] and c["text"]: add(c, [s], evs.get(f"{i}|{s}"))
    for c in new.get(i, {}).get("candidates", []): add(c, ["rl_latest_gen"], c.get("ev"), greedy=c["greedy"])
    for c in cands: c.pop("_g")
    fl = {s: flash[f"{i}|{s}"] for s in ("reference", "sft", "rl_last") if f"{i}|{s}" in flash}
    for c in cands:
        c["flash_overall"] = {s: fl[s].get("overall") for s in c["systems"] if s in fl}
        if c["ev"] is None: c["ev"] = None
    if len(cands) >= 2: evalrows.append({"id": i, "origin": "test_set", "task": r["task"], "lang": r["lang"], "switch": r.get("switch"), "english": r["prompt"].split(":", 1)[1].strip(), "prompt": r["prompt"], "candidates": cands})
print("test-set items", len(evalrows), "merged duplicate candidates", dupes)

# ---- selection
langs = sorted({r["lang"] for r in evalrows} | {r["lang"] for r in pool}); ecount = collections.Counter(r["lang"] for r in evalrows)
usable = [r for r in pool if len(r["candidates"]) >= 4]; pavail = collections.Counter(r["lang"] for r in usable)
def total(c): return sum(max(ecount[l], min(c, ecount[l] + pavail[l])) for l in langs)
lo, hi = 0, 5000
while lo < hi:
    mid = (lo + hi + 1) // 2
    if total(mid) <= TARGET: lo = mid
    else: hi = mid - 1
level = lo; need = {l: max(0, min(level, ecount[l] + pavail[l]) - ecount[l]) for l in langs}
chosen = []; taken = set()
def spread_pick(rows, n):
    """rows sorted by mean evaluator score, take n evenly spaced (covers the whole score range)"""
    if n >= len(rows): return rows
    rows = sorted(rows, key=lambda r: mean([c["ev"] for c in r["candidates"]]))
    return [rows[int((j + 0.5) * len(rows) / n)] for j in range(n)]
for l in langs:
    n = need[l]
    if not n: continue
    by = collections.defaultdict(list)
    for r in usable:
        if r["lang"] == l: by[r["task"]].append(r)
    tasks = sorted(by); quota = {t: n // len(tasks) for t in tasks}
    for t in tasks[: n - sum(quota.values())]: quota[t] += 1
    short = 0
    for t in tasks:
        q = min(quota[t], len(by[t])); short += quota[t] - q; chosen += spread_pick(by[t], q)
    if short:                                                               # a task ran out in this language: fill from the others, stratified
        left = [r for t in tasks for r in by[t] if r not in chosen]; chosen += spread_pick(left, short)
left = TARGET - len(evalrows) - len(chosen)                                   # rounding slack: stratified over the rest
if left > 0:
    ids = {r["id"] + r["task"] for r in chosen}; rest = [r for r in usable if r["id"] + r["task"] not in ids]
    chosen += spread_pick(rest, left)
items = evalrows + chosen
print("level", level, "chosen from pool", len(chosen), "total", len(items))

# ---- pairs: greedy vs sampled, widest and narrow evaluator gap, gold vs best system
def pair(a, b, kind, r):
    return {"item": r["id"] + "|" + r["task"], "kind": kind, "a": a, "b": b, "ev_gap": None if a["ev"] is None or b["ev"] is None else round(abs(a["ev"] - b["ev"]), 4), "swap": rng.random() < 0.5}
pairs = []
for r in items:
    cs = r["candidates"]; seen = set(); P = []
    def put(a, b, kind):
        key = tuple(sorted((a["text"], b["text"])))
        if a is b or key in seen: return
        seen.add(key); P.append(pair({k: a[k] for k in ("text", "systems", "ev", "greedy")}, {k: b[k] for k in ("text", "systems", "ev", "greedy")}, kind, r))
    gr = [c for c in cs if c["greedy"] and c["ev"] is not None]; sm = [c for c in cs if not c["greedy"] and c["ev"] is not None and ("rl_latest_gen" in c["systems"] or "gen" in c["systems"])]
    scored = [c for c in cs if c["ev"] is not None]
    if gr and sm: put(gr[0], rng.choice(sm), "greedy_vs_sampled")
    if len(scored) >= 2: put(max(scored, key=lambda c: c["ev"]), min(scored, key=lambda c: c["ev"]), "widest_evaluator_gap")
    if len(scored) >= 4:
        s = sorted(scored, key=lambda c: c["ev"]); m = len(s) // 2; put(s[m - 1], s[m], "adjacent_evaluator")
    refc = [c for c in cs if "reference" in c["systems"]]; bst = [c for c in cs if "rl_last" in c["systems"]]
    if refc and bst: put(refc[0], bst[0], "reference_vs_rl_last")
    pairs += P
# attention checks: a candidate against a word-shuffled copy of itself (should always lose)
checks = []; byl = collections.defaultdict(list)
for r in items: byl[r["lang"]].append(r)
for l, rs in byl.items():
    for r in rng.sample(rs, min(len(rs), max(6, len(rs) // 40))):
        c = max(r["candidates"], key=lambda c: len(c["text"]))
        if len(c["text"].split()) < 6: continue
        w = c["text"].split(); sh = w[:]
        for _ in range(10):
            rng.shuffle(sh)
            if sh != w: break
        checks.append({"item": r["id"] + "|" + r["task"], "lang": l, "kind": "attention_shuffled", "good": c["text"], "bad": " ".join(sh), "swap": rng.random() < 0.5})

# ---- outputs
def dump(p, rows):
    with (gzip.open(p, "wt", encoding="utf-8") if p.endswith(".gz") else open(p, "w", encoding="utf-8")) as fo:
        for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
dump(f"{OUT}/test_set/items.jsonl.gz", items); dump(f"{OUT}/test_set/pairs.jsonl.gz", pairs); dump(f"{OUT}/test_set/attention_checks.jsonl", checks)
plang = collections.Counter({}); lang_of = {r["id"] + "|" + r["task"]: r["lang"] for r in items}
for p in pairs: plang[lang_of[p["item"]]] += 1
S = {"target": TARGET, "items": len(items), "test_set_items": len(evalrows), "pool_items": sum(r["origin"] == "pool" for r in chosen), "test_set_eng_m_items": sum(r["origin"] == "test_set_eng_m" for r in chosen), "water_level_per_language": level, "pairs": len(pairs), "attention_checks": len(checks),
     "pairs_by_kind": dict(collections.Counter(p["kind"] for p in pairs)), "items_by_task": dict(collections.Counter(r["task"] for r in items)),
     "items_per_language": {l: {"test_set": ecount[l], "pool": sum(1 for r in chosen if r["lang"] == l), "total": ecount[l] + sum(1 for r in chosen if r["lang"] == l), "pairs": plang[l]} for l in langs},
     "candidates_per_item": round(mean([len(r["candidates"]) for r in items]), 2),
     "evaluator_mean_by_origin": {o: round(mean([c["ev"] for r in items if r["origin"] == o for c in r["candidates"] if c["ev"] is not None]), 3) for o in ("test_set", "pool", "test_set_eng_m") if any(r["origin"] == o for r in items)}}
evall = [c["ev"] for r in items for c in r["candidates"] if c["ev"] is not None]; q = statistics.quantiles(evall, n=10)
S["evaluator_deciles"] = [round(x, 2) for x in q]
S["pair_ev_gap_median"] = round(statistics.median([p["ev_gap"] for p in pairs if p["ev_gap"] is not None]), 3)
json.dump(S, open(f"{OUT}/test_set/stats.json", "w"), indent=1)
print(json.dumps({k: v for k, v in S.items() if k != "items_per_language"}, indent=1)); print(S["items_per_language"])
