"""Dedupe the pool, then score every candidate with the group-token evaluator (final text + devices, mean of the task's dimensions, 1-5).
python score_all.py EVALUATOR_DIR OUT_DIR ROWFILE... ; ROWFILE = jsonl or jsonl.gz in the gen_main/postprocess schema (task m_text|hm|ood, optional task6).
Pool dedupe: one row per (task, normalised English); rows whose English appears in the test-set prompts are dropped."""
import sys, json, gzip, re, unicodedata, collections, os, time
sys.path.insert(0, "/workspace/csw_hs/src")
from csw.rl.evaluator import EvaluatorScorer
evdir, outdir, files = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(outdir, exist_ok=True)
opn = lambda p: gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
norm = lambda t: " ".join(re.sub(r"[^\w ]", "", unicodedata.normalize("NFC", t).lower()).split())
TASK6 = {"m_text": "eng_m", "hm": "eng_hm", "ood": "eng_ood"}
test_en = {norm(json.loads(l)["prompt"].split(":", 1)[1]) for l in open("/workspace/csw_hs/data/test_off.jsonl")}
sc = EvaluatorScorer(evdir, bs=256)
for f in files:
    name = os.path.basename(f).split(".")[0]; rows = [json.loads(l) for l in opn(f)]; seen = set(); keep = []; stat = collections.Counter()
    for r in rows:
        k = (r["task"], norm(r["english"]))
        if "task6" not in r:
            if k in seen: stat["dup_english"] += 1; continue
            if norm(r["english"]) in test_en: stat["in_test_set"] += 1; continue
            seen.add(k)
        if not r["candidates"]: stat["empty"] += 1; continue
        keep.append(r)
    items = [(r.get("task6") or TASK6[r["task"]], r["lang"], r.get("switch") or "", r["english"], c["text"], c["devices"]) for r in keep for c in r["candidates"]]
    t0 = time.time(); out = []
    for i in range(0, len(items), 20000):
        out += sc(items[i:i + 20000]); print(name, i + 20000, "/", len(items), f"{time.time() - t0:.0f}s", flush=True)
    j = 0
    for r in keep:
        for c in r["candidates"]: c["ev"] = round(out[j], 4); j += 1
    with gzip.open(f"{outdir}/{name}.jsonl.gz", "wt", encoding="utf-8") as fo:
        for r in keep: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, "rows in", len(rows), "kept", len(keep), dict(stat), "candidates scored", len(items), f"{time.time() - t0:.0f}s", flush=True)
