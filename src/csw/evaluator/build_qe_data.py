"""Judged rows -> quality-estimator (QE) training data.
python -m csw.evaluator.build_qe_data --out work/qe_data
One record per judged row holds the four text fields, the plan / idiom devices and a 27-dim label vector:
  0-5 M1..M6 | 6-14 H1..H9 | 15-21 O1..O7 | 22 D1 | 23 D2 | 24 X1 | 25 pd_mean (plan items, 0-2 -> 1-5 scale) | 26 iq_mean (idioms)
A label is the mean of the judges that graded it (judge scales are first matched: luna is harsher, so its grades are mapped onto Flash's scale per dimension using the rows both judges graded).
Split: the pool's eval rows (judged by BOTH judges, plus the original/repaired pairs) are the test set; train / val come from every other judged row, split by English sentence."""
from csw.paths import DATA_ROOT
import argparse, collections, hashlib, json, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.sft import devices as dv

DIMS = [f"M{i}" for i in range(1, 7)] + [f"H{i}" for i in range(1, 10)] + [f"O{i}" for i in range(1, 8)] + ["D1", "D2", "X1", "PD", "IQ"]


def grade_vec(rep):
    g = rep.get("g")
    if not isinstance(g, list) or len(g) != 5 or not all(isinstance(a, list) for a in g): return None
    v = [None] * 27

    def put(i, x):
        if isinstance(x, (int, float)) and not isinstance(x, bool) and 1 <= x <= 5: v[i] = float(x)
    for i in range(6): put(i, g[0][i] if len(g[0]) > i else None)
    for i in range(9): put(6 + i, g[1][i] if len(g[1]) > i else None)
    for i in range(7): put(15 + i, g[2][i] if len(g[2]) > i else None)
    put(22, g[3][0] if len(g[3]) > 0 else None); put(23, g[3][1] if len(g[3]) > 1 else None); put(24, g[4][0] if len(g[4]) > 0 else None)
    for k, i in (("pd", 25), ("iq", 26)):
        a = [x for x in (rep.get(k) or []) if isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 2]
        if a: v[i] = 1 + 2 * sum(a) / len(a)                           # 0..2 -> 1..5
    if rep.get("r") == "n": return None
    return v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="work/qe_data"); ap.add_argument("--rows", default="work/post_rows"); a = ap.parse_args()
    runs = [("flash", "gemini-3.8-flash", "pool_flash"), ("luna", "gpt-5.6-luna", "pool_luna"), ("flash", "gemini-3.8-flash", "pool_r2_flash"), ("luna", "gpt-5.6-luna", "pool_r2_luna")]
    pool = {}
    for f in ("work/judge_pool_v2/judge_pool.jsonl", "work/judge_pool_r2/judge_pool.jsonl"):
        for l in open(f):
            p = json.loads(l); pool[p["id"]] = p
    labels = collections.defaultdict(dict)                          # id -> judge -> vec
    meta = {}
    for judge, jn, d in runs:
        fn = Path(f"work/{d}/results_{jn}@v13.jsonl")
        if not fn.exists(): print("missing", fn); continue
        for l in open(fn):
            r = json.loads(l); rep = r.get("reply")
            if not rep: continue
            v = grade_vec(rep)
            if v and r["rid"] not in labels or (v and judge not in labels[r["rid"]]): labels[r["rid"]][judge] = v
    # scale matching luna -> flash on the rows both graded
    st = {}
    for i in range(27):
        xs = [(l["luna"][i], l["flash"][i]) for l in labels.values() if "luna" in l and "flash" in l and l["luna"][i] is not None and l["flash"][i] is not None]
        if len(xs) > 50:
            ml = sum(x for x, _ in xs) / len(xs); mf = sum(y for _, y in xs) / len(xs)
            sl = (sum((x - ml) ** 2 for x, _ in xs) / len(xs)) ** .5 or 1; sf = (sum((y - mf) ** 2 for _, y in xs) / len(xs)) ** .5 or 1
            st[i] = (ml, sl, mf, sf)
    json.dump({DIMS[i]: v for i, v in st.items()}, open("work/qe_luna_to_flash.json", "w"))
    rows = {}
    for f in sorted(Path(a.rows).glob("*.raw.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in labels: rows[r["id"]] = r
    lex = dv.Lexicon.load("work/lexicon.json")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n = collections.Counter()
    fo = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    for rid, lab in labels.items():
        r = rows.get(rid)
        if not r or not all(isinstance(r.get(k), str) for k in ("eng", "m_text", "hm", "ood")): continue
        p = pool.get(rid, {}); vec = []
        for i in range(27):
            f = lab["flash"][i] if "flash" in lab else None; x = lab["luna"][i] if "luna" in lab else None
            if x is not None and i in st: ml, sl, mf, sf = st[i]; x = min(5.0, max(1.0, mf + (x - ml) / sl * sf))
            vec.append(round(0.7 * f + 0.3 * x, 3) if f is not None and x is not None else (round(f, 3) if f is not None else (round(x, 3) if x is not None else None)))   # Flash 0.7 : luna 0.3 when both graded
        try: plan = dv.hm_devices(r, lex)["devices"]
        except Exception: plan = []
        idioms = [d for d in (r.get("devices") or []) if isinstance(d, str)][:8]
        sha = hashlib.sha1(" ".join(r["eng"].lower().split()).encode()).hexdigest()
        rec = {"id": rid, "lang": r["lang"], "pin": (r.get("scenario") or {}).get("switch") or "", "eng": r["eng"], "m": r["m_text"], "hm": r["hm"], "ood": r["ood"],
               "plan": plan[:10], "idioms": idioms, "draft": " ".join((r.get("eng_initial") or "").split()), "y": vec, "yj": lab, "judges": sorted(lab), "pair": p.get("pair"), "tier": p.get("tier"), "sha": sha}
        if p.get("split") == "eval": s = "test"
        else: s = "val" if int(sha[:4], 16) % 50 == 0 else "train"
        fo[s].write(json.dumps(rec, ensure_ascii=False) + "\n"); n[s] += 1
    for f in fo.values(): f.close()
    print(dict(n), "luna->flash scale on", len(st), "dims")


if __name__ == "__main__":
    main()
