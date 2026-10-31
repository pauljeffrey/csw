"""Pilot 2 of the judge card: fresh natural rows + graded corruption ladders, scored by Gemini (direct API) and gpt-4o, then compared.

python -m csw.judging.judge_pilot2 build  --pool ... --rows ... --lexicon ... --langs yor tum --exclude-ids ids.txt --out DIR
python -m csw.judging.judge_pilot2 run    --dir DIR --judge gemini-3.8-flash | gpt-4o
python -m csw.judging.judge_pilot2 report --dir DIR --judges gemini-3.8-flash gpt-4o
Nothing is sent by `build`.  `run` resumes from <dir>/results_<judge>.jsonl.
"""
from __future__ import annotations

from csw.paths import DATA_ROOT, env_value
import argparse
import collections
import json
import os
import random
import statistics as st
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = DATA_ROOT
from csw.sft import devices as dv   # noqa: E402
from csw.judging import corrupt_ladder as L   # noqa: E402
from csw.judging import judge_v11_test as J   # noqa: E402
from csw.sft.apply_edits import apply_edits                                   # noqa: E402

LADDER_OPS = {"yor": ["H5_overmix", "H4_thirdlang", "O2_idiom_literal", "M3_diacritics"], "default": ["H5_overmix", "H4_thirdlang", "O2_idiom_literal", "M4_english"]}
LEVELS_USED = (1, 3)


def norm_reply(y):
    """compact v1.3 replies (arrays) -> the dict layout of v1.2 so one report handles both"""
    if not isinstance(y, dict):
        return y
    y = dict(y)
    if isinstance(y.get("g"), list):
        y["g"] = dict(zip("mhodx", y["g"]))
    if y.get("e") and isinstance(y["e"][0], list):
        y["e"] = [{"f": e[0], "s": e[1], "c": e[2], "v": e[3], "d": e[4], "n": ""} for e in y["e"]]
    if y.get("ed") and isinstance(y["ed"][0], list):
        y["ed"] = [{"f": e[0], "find": e[1], "ctx": "", "rep": e[2], "n": ""} for e in y["ed"]]
    if "fx" in y and "fix" not in y:
        y["fix"] = y["fx"]
    return y


def _g(x, k, i):
    try:
        return x["reply"]["g"][k][i]
    except (KeyError, IndexError, TypeError):
        return None


def dim_index(d: str):
    return {"M": "m", "H": "h", "O": "o", "D": "d", "X": "x"}[d[0]], int(d[1:]) - 1


def build(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    used = set(open(a.exclude_ids).read().split()) if a.exclude_ids else set()
    pool = [json.loads(l) for l in open(a.pool, encoding="utf-8") if l.strip()]
    lex = dv.Lexicon.load(a.lexicon); rnd = random.Random(a.seed)
    rows, ctx_pool = {}, {}
    for lang in a.langs:
        rs = [json.loads(l) for l in open(Path(a.rows) / f"{lang}.raw.jsonl", encoding="utf-8") if l.strip()]
        rs = [r for r in rs if all(r.get(k) for k in ("eng", "m_text", "hm", "ood"))]
        ctx_pool[lang] = rs[:600]
        for r in rs:
            rows[r["id"]] = r
    foreign_other = {}
    for lang in a.langs:
        toks = []
        for k, v in lex.foreign.items():
            lg, sw = k.split("|")
            if lg != lang and sw != "English":
                toks += sorted(v)[:300]
        foreign_other[lang] = toks
    ctx = {"pool_by_lang": ctx_pool, "lexicon": lex, "foreign_other": foreign_other}
    reqs = []
    for lang in a.langs:
        ev = [p for p in pool if p["lang"] == lang and p["split"] == "eval" and p["id"] not in used and p["id"] in rows]
        rnd.shuffle(ev)
        nat = collections.defaultdict(list)
        for p in ev:
            if len(nat[p["tier"]]) < a.natural_per_tier:
                nat[p["tier"]].append(p)
        for tier, ps in nat.items():
            for p in ps:
                reqs.append({"rid": p["id"], "kind": "natural", "lang": lang, "tier": tier, "row": rows[p["id"]]})
        base = [p for p in ev if p["tier"] in ("good", "medium") and p["id"] not in {q["id"] for v in nat.values() for q in v}][: a.ladder_bases]
        ops = a.ops or LADDER_OPS.get(lang, LADDER_OPS["default"])
        for p in base:
            r = rows[p["id"]]
            reqs.append({"rid": p["id"] + "#L0", "kind": "ladder", "lang": lang, "tier": p["tier"], "op": "orig", "level": 0, "base": p["id"], "dim": None, "row": r})
            for op in ops:
                for ch in L.make_chain(r, op, rnd, ctx):
                    if ch["level"] in (tuple(a.levels) if a.levels else LEVELS_USED):
                        v = ch["row"]; v["id"] = f"{p['id']}#{op}{ch['level']}"
                        reqs.append({"rid": v["id"], "kind": "ladder", "lang": lang, "tier": p["tier"], "op": op, "level": ch["level"], "base": p["id"], "dim": ch["meta"]["dim"], "row": v})
    with open(out / "requests.jsonl", "w", encoding="utf-8") as f:
        for q in reqs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(len(reqs), "requests:", dict(collections.Counter((q["lang"], q["kind"]) for q in reqs)))


def call_openai(model: str, system: str, user: str, key: str) -> dict:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "response_format": {"type": "json_object"}}
    if model.startswith("gpt-5") or model.startswith("gpt-6"):           # reasoning models: no temperature, max_completion_tokens, reasoning effort
        body["max_completion_tokens"] = 8000; body["reasoning_effort"] = "low"
    else:
        body["temperature"] = 0; body["max_tokens"] = 3500
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            o = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    return {"text": o["choices"][0]["message"]["content"], "usage": {"in": o["usage"]["prompt_tokens"], "out": o["usage"]["completion_tokens"]}, "sec": round(time.time() - t0, 1)}


def openai_key() -> str:
    k = env_value("OPENAI_API_KEY")
    return k


def run(a):
    d = Path(a.dir); reqs = [json.loads(l) for l in open(d / "requests.jsonl", encoding="utf-8")]
    lex = dv.Lexicon.load(a.lexicon); system, schema = J.card_parts(a.card)
    spec = a.judge if a.card == "v11" else f"{a.judge}@{a.card}"
    res_f = d / f"results_{spec}.jsonl"; done = set()
    if res_f.exists():
        done = {json.loads(l)["rid"] for l in open(res_f, encoding="utf-8") if l.strip()}
    todo = [q for q in reqs if q["rid"] not in done]
    lock = threading.Lock(); fo = open(res_f, "a", encoding="utf-8"); k = J.key() if a.judge.startswith("gemini") else openai_key(); tot = collections.Counter()
    try:
        import jsonschema
        val = jsonschema.Draft7Validator(schema)
    except ImportError:
        val = None

    gate = {"t": 0.0}

    def one(q):
        if a.min_interval:                                   # free-tier keys allow only 5 requests a minute: space the calls out
            with lock:
                wait = gate["t"] + a.min_interval - time.time()
                gate["t"] = max(time.time(), gate["t"] + a.min_interval)
            if wait > 0:
                time.sleep(wait)
        r = q["row"]; user = J.user_message(r, lex, compact=(a.card == "v13"))
        if a.judge.startswith("gemini"):
            resp = J.call(a.judge, system, user, k, "low", 9000 if "pro" in a.judge else 4000)
            if "error" in resp:
                out = {"error": resp["error"]}
            else:
                um = resp.get("usageMetadata", {}); text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"] if "text" in p)
                out = {"text": text, "usage": {"in": um.get("promptTokenCount", 0), "out": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)}, "sec": resp.get("_sec")}
        else:
            out = call_openai(a.judge, system, user, k)
        rec = {"rid": q["rid"], "judge": spec, "error": out.get("error"), "usage": out.get("usage"), "sec": out.get("sec")}
        if "text" in out:
            try:
                rec["reply"] = json.loads(out["text"])
                rec["schema_errors"] = [e.message[:80] for e in val.iter_errors(rec["reply"])] if val else []
            except Exception:
                rec["reply"] = None; rec["raw"] = out["text"][:500]
        if a.card == "v13" and isinstance(rec.get("reply"), dict) and not rec["reply"].get("nn"):          # repair pass: a field with a major/critical span but no fx/ed gets ONE follow-up call
            y = rec["reply"]; miss = sorted({e[0] for e in y.get("e") or [] if isinstance(e, list) and len(e) >= 4 and e[3] in ("j", "c")} - {kk for kk, vv in (y.get("fx") or {}).items() if vv} - {e[0] for e in y.get("ed") or [] if isinstance(e, list) and e})
            if miss:
                fu = user + "\n\nYour reply listed major or critical errors in field(s) " + ",".join(miss) + " but gave no correction for them. Reply ONLY with JSON {\"fx\":{...}} or {\"ed\":[...]} for those fields (or {\"nn\":1} if you cannot fix them without inventing words). Your errors: " + json.dumps([e for e in y["e"] if e[0] in miss], ensure_ascii=False)
                o2 = J.call(a.judge, system, fu, k, "low", 4000) if a.judge.startswith("gemini") else call_openai(a.judge, system, fu, k)
                try:
                    if "text" not in o2 and "candidates" in o2:
                        o2 = {"text": "".join(p_.get("text", "") for p_ in o2["candidates"][0]["content"]["parts"] if "text" in p_), "usage": {"in": o2.get("usageMetadata", {}).get("promptTokenCount", 0), "out": o2.get("usageMetadata", {}).get("candidatesTokenCount", 0) + o2.get("usageMetadata", {}).get("thoughtsTokenCount", 0)}}
                    r2 = json.loads(o2["text"]); rec["repair"] = {"fields": miss, "reply": r2}
                    for kk in ("fx", "ed", "nn"):
                        if r2.get(kk):
                            y[kk] = (y.get(kk) or ([] if kk == "ed" else {})) + r2[kk] if kk == "ed" else ({**(y.get(kk) or {}), **r2[kk]} if kk == "fx" else r2[kk])
                    if o2.get("usage"):
                        out["usage"] = {"in": (out.get("usage") or {}).get("in", 0) + o2["usage"]["in"], "out": (out.get("usage") or {}).get("out", 0) + o2["usage"]["out"]}; rec["usage"] = out["usage"]
                except Exception:
                    rec["repair"] = {"fields": miss, "reply": None}
        with lock:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
            if out.get("usage"):
                tot["in"] += out["usage"]["in"]; tot["out"] += out["usage"]["out"]
            tot["n"] += 1
            if tot["n"] % 10 == 0:
                print(a.judge, tot["n"], "/", len(todo), flush=True)

    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(one, todo))
    price = {"gpt-4o": (1.25, 5.0), "gpt-5.5": (2.5, 15.0), "gemini-3.1-pro-preview": (1.0, 6.0), "gpt-5.6-luna": (0.10, 0.60), "gpt-5.4-mini": (0.375, 2.25), "gpt-5.6-terra": (1.0, 6.0)}.get(a.judge, (0.375, 1.875))                 # batch prices; realtime is about 2x
    print(f"{spec}: {tot['n']} calls, tokens in {tot['in']} out {tot['out']}, batch-price cost ${tot['in'] * price[0] / 1e6 + tot['out'] * price[1] / 1e6:.3f} (realtime about 2x)")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def report(a):
    d = Path(a.dir); reqs = {q["rid"]: q for q in map(json.loads, open(d / "requests.jsonl", encoding="utf-8"))}
    R = {}
    for j in a.judges:
        f = d / f"results_{j}.jsonl"
        R[j] = {x["rid"]: dict(x, reply=norm_reply(x["reply"])) for x in map(json.loads, open(f, encoding="utf-8")) if x.get("reply")} if f.exists() else {}
    for j, rr in R.items():
        tot = len([1 for _ in open(d / f"results_{j}.jsonl")]) if (d / f"results_{j}.jsonl").exists() else 0
        sch = sum(1 for x in rr.values() if x.get("schema_errors"))
        print(f"{j}: {len(rr)}/{tot} parsed, {sch} with schema errors")
        # edits and fix policy
        n_ed = ok_ed = missing_fix = 0
        for rid, x in rr.items():
            y = x["reply"]; q = reqs[rid]; row = q["row"]
            fld = {"m": row["m_text"], "h": row["hm"], "o": row["ood"]}
            for e in y.get("ed") or []:
                n_ed += 1; ok_ed += apply_edits(fld[e["f"]], [e])[1]["ok"]
            sev = collections.defaultdict(set)
            for e in y.get("e") or []:
                if e["v"] in ("j", "c"):
                    sev[e["f"]].add(e["d"])
            for f_ in sev:
                has_ed = any(e["f"] == f_ for e in y.get("ed") or [])
                if not ((y.get("fix") or {}).get(f_) or has_ed) and not y.get("nn"):
                    missing_fix += 1
        print(f"   edit spans: {n_ed} given, {ok_ed} apply cleanly; fields with a major/critical span but no fix/edit: {missing_fix}")
        ks = [x["reply"].get("k") for x in rr.values() if isinstance(x["reply"].get("k"), (int, float))]
        print(f"   confidence k: mean {st.mean(ks):.2f}, min {min(ks):.2f}, max {max(ks):.2f}")
    # ladder: grade of the damaged dimension by level
    print("\nLADDER: mean grade of the damaged dimension by damage level (0 = original; lower at higher level = the judge sees the damage)")
    for j, rr in R.items():
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for rid, x in rr.items():
            q = reqs[rid]
            if q["kind"] != "ladder":
                continue
            base = q["base"]
            if q["op"] == "orig":
                continue
            k, i = dim_index(q["dim"]); g = _g(x, k, i)
            orig = R[j].get(base + "#L0")
            g0 = _g(orig, k, i) if orig else None
            if g is not None:
                by[(q["lang"], q["op"])][q["level"]].append(g)
                if g0 is not None:
                    by[(q["lang"], q["op"])][0].append(g0)
        print(f"  [{j}]")
        for (lang, op), lv in sorted(by.items()):
            print("   %-4s %-18s" % (lang, op) + "  ".join(f"L{l}: {st.mean(v):.2f} (n={len(v)})" for l, v in sorted(lv.items())))
    import itertools
    names = {"m": 6, "h": 9, "o": 7, "d": 2, "x": 3}
    print("\nPAIRWISE AGREEMENT over all graded dimensions (pooled Spearman, n pairs of grades) and mean grade per judge")
    for j in R:
        vals = [g for x in R[j].values() for k, n in names.items() for g in ((x["reply"].get("g") or {}).get(k) or []) if g is not None]
        print(f"   {j:24} mean grade {st.mean(vals):.2f}  share of grades equal to 5: {sum(v == 5 for v in vals) / len(vals):.2f}  share <=2: {sum(v <= 2 for v in vals) / len(vals):.2f}")
    for j1, j2 in itertools.combinations(R, 2):
        both = [rid for rid in R[j1] if rid in R[j2]]; X, Y = [], []
        per = collections.defaultdict(lambda: ([], []))
        for rid in both:
            for k, n in names.items():
                for i2 in range(n):
                    g1, g2 = _g(R[j1][rid], k, i2), _g(R[j2][rid], k, i2)
                    if g1 is not None and g2 is not None:
                        X.append(g1); Y.append(g2); per[f"{k.upper()}{i2 + 1}"][0].append(g1); per[f"{k.upper()}{i2 + 1}"][1].append(g2)
        good = {d: round(spearman(x, y), 2) for d, (x, y) in per.items() if len(x) >= 20 and len(set(x)) > 1 and len(set(y)) > 1}
        print(f"   {j1} vs {j2}: n={len(X)} pooled spearman {spearman(X, Y):.2f}; per-dimension (n>=20): {good}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--pool", required=True); b.add_argument("--rows", required=True); b.add_argument("--lexicon", required=True); b.add_argument("--langs", nargs="+", required=True)
    b.add_argument("--exclude-ids"); b.add_argument("--out", required=True); b.add_argument("--natural-per-tier", type=int, default=2); b.add_argument("--ladder-bases", type=int, default=3); b.add_argument("--seed", type=int, default=7); b.add_argument("--ops", nargs="+"); b.add_argument("--levels", nargs="+", type=int)
    r = sp.add_parser("run"); r.add_argument("--dir", required=True); r.add_argument("--judge", required=True); r.add_argument("--lexicon", required=True); r.add_argument("--conc", type=int, default=6); r.add_argument("--min-interval", type=float, default=0.0); r.add_argument("--card", default="v12", choices=["v11", "v12", "v13"])
    p = sp.add_parser("report"); p.add_argument("--dir", required=True); p.add_argument("--judges", nargs="+", required=True)
    a = ap.parse_args()
    {"build": build, "run": run, "report": report}[a.cmd](a)
