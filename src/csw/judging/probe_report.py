"""Capability screen + repeatability for the judges.  python -m csw.judging.probe_report DIR1 DIR2   (two runs of the same requests)"""
import collections, json, statistics as st, sys
from csw.judging.judge_pilot2 import spearman
D1, D2 = sys.argv[1], sys.argv[2]
req = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{D1}/requests.jsonl")}
J = {"gemini-3.8-flash": "flash", "gpt-5.6-luna": "luna"}
def load(d, j):
    return {x["rid"]: x["reply"] for x in map(json.loads, open(f"{d}/results_{j}@v13.jsonl")) if isinstance(x.get("reply"), dict) and isinstance(x["reply"].get("g"), list)}
def g(y, k, i):
    try:
        v = y["g"][k][i]; return v if isinstance(v, int) else None
    except Exception:
        return None
R = {j: (load(D1, j), load(D2, j)) for j in J}
print("== REPEATABILITY (same judge, same rows, run 1 vs run 2) ==")
for j, (a, b) in R.items():
    X, Y = [], []; per = collections.defaultdict(lambda: ([], []))
    for r in set(a) & set(b):
        for k, n in enumerate([6, 9, 7, 2, 3]):
            for i in range(n):
                u, v = g(a[r], k, i), g(b[r], k, i)
                if u is not None and v is not None:
                    X.append(u); Y.append(v); per["MHODX"[k] + str(i + 1)][0].append(u); per["MHODX"[k] + str(i + 1)][1].append(v)
    ex = sum(u == v for u, v in zip(X, Y)) / len(X); w1 = sum(abs(u - v) <= 1 for u, v in zip(X, Y)) / len(X)
    lv = sum(a[r].get("l") != b[r].get("l") for r in set(a) & set(b)) / len(set(a) & set(b))
    nn = sum(bool(a[r].get("nn")) != bool(b[r].get("nn")) for r in set(a) & set(b)) / len(set(a) & set(b))
    low = sorted(((round(spearman(x, y), 2), d) for d, (x, y) in per.items() if len(x) > 40 and len(set(x)) > 1 and len(set(y)) > 1))[:6]
    print(f"{J[j]}: rows {len(set(a) & set(b))}, grades {len(X)}, spearman {spearman(X, Y):.2f}, exact {ex:.2f}, within-1 {w1:.2f}, mean {st.mean(X):.2f} vs {st.mean(Y):.2f}, reading level changed on {lv:.0%} of rows, nn flips {nn:.0%}; least repeatable dims {low}")
print("\n== CAPABILITY SCREEN per language (both runs pooled: 4 original rows, 4 English-injected, 4 third-language) ==")
print("drop = mean(original grade - damaged grade) on M4 (English injected) and H4 (third language); limited = share of rows where the judge says its level is 'limited'")
res = {}
for j, (a, b) in R.items():
    for lang in sorted({q["lang"] for q in req.values()}):
        dE, dT, lim, n = [], [], 0, 0
        for run in (a, b):
            for rid, q in req.items():
                if q["lang"] != lang or rid not in run: continue
                y = run[rid]; n += 1; lim += y.get("l") == "l" or y.get("r") == "n"
                if q["op"] in ("M4_english", "H4_thirdlang"):
                    orig = run.get(q["base"] + "#L0")
                    k, i = (0, 3) if q["op"] == "M4_english" else (1, 3)
                    if orig is not None and g(orig, k, i) is not None and g(y, k, i) is not None:
                        (dE if q["op"] == "M4_english" else dT).append(g(orig, k, i) - g(y, k, i))
        res[(j, lang)] = (st.mean(dE) if dE else None, st.mean(dT) if dT else None, lim / max(1, n), len(dE), len(dT))
def verdict(t):
    e, h, lim = t[0], t[1], t[2]
    if e is None or h is None: return "?"
    ok = (e >= 1) + (h >= 1)
    if lim >= 1 / 3 or ok == 0: return "EXCLUDE"
    return "ok" if ok == 2 else "partial"
print("lang   | flash: dEng dThird lim verdict | luna: dEng dThird lim verdict")
ex = {j: [] for j in J}
for lang in sorted({q["lang"] for q in req.values()}):
    line = f"{lang:5} |"
    for j in J:
        t = res[(j, lang)]; v = verdict(t)
        line += (f" {t[0] if t[0] is None else round(t[0], 1):>5} {t[1] if t[1] is None else round(t[1], 1):>5} {t[2]:.2f} {v:8} |")
        if v == "EXCLUDE": ex[j].append(lang)
    print(line)
for j in J: print(J[j], "exclude:", ex[j], "| partial:", [l for l in {q['lang'] for q in req.values()} if verdict(res[(j, l)]) == "partial"])
