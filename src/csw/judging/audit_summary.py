"""Per-language summary of the Gemini gold audit. python -m csw.judging.audit_summary"""
from csw.paths import DATA_ROOT
import json, collections, statistics as st, sys
from pathlib import Path
HERE = DATA_ROOT; sys.path.insert(0, str(HERE))
from csw.datagen import seeds as sm
from csw.datagen import gates
FILES = ("gold_audit.pilot", "gold_audit_flagged", "gold_audit_rest", "gold_audit_retry")
rows = []
for f in FILES:
    p = HERE / "exemplars" / f"{f}.gemini.jsonl"
    if p.exists():
        for l in p.read_text(encoding="utf-8").split("\n"):
            if l.strip(): rows.append(json.loads(l))
best = {}
for r in rows:
    k = (r["lang"], r["eng"], r["m_text"], r["hm"], r["ood"])
    if k not in best or best[k]["proposal"].get("verdict") in (None, "error"): best[k] = r     # a later good result replaces an error
rows = list(best.values())
S = lambda x: x if isinstance(x, str) else ""
N = lambda x: x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
D = lambda r, k: r["proposal"].get(k) if isinstance(r["proposal"].get(k), dict) else {}
ok = [r for r in rows if r["proposal"].get("verdict") in ("ok", "fixed", "partial", "unsalvageable") and all(isinstance(r["proposal"].get(k), str) for k in ("m_text", "hm", "ood"))]
print(len(rows), "unique rows;", len(ok), "valid proposals;", len(rows) - len(ok), "errors")
vocab = {}
def voc(l):
    if l not in vocab: vocab[l] = {w for s in sm.load(l) for w in gates.words(s["text"])}
    return vocab[l]
def oov(t, l):
    ws = [w for w in gates.words(S(t)) if w not in gates.EN]; v = voc(l)
    return sum(w not in v for w in ws) / max(1, len(ws)) if v and ws else None
mean = lambda xs: (lambda y: round(st.mean(y), 1) if y else "-")([x for x in xs if x is not None])
by = collections.defaultdict(list)
for r in ok: by[r["lang"]].append(r)
out = []
for l, rs in by.items():
    vc = collections.Counter(r["proposal"]["verdict"] for r in rs); lev = collections.Counter(D(r, "language_proficiency").get("level") for r in rs)
    pre = [mean([N(D(r, "scores_before").get(k)) for r in rs]) for k in ("m_text", "hm", "ood")]; post = [mean([N(D(r, "scores_after").get(k)) for r in rs]) for k in ("m_text", "hm", "ood")]
    a = [x for x in (oov(r["m_text"], l) for r in rs) if x is not None]; b = [x for x in (oov(r["proposal"]["m_text"], l) for r in rs) if x is not None]
    out.append((pre[0] if isinstance(pre[0], float) else 99, l, len(rs), vc["fixed"] + vc["partial"], vc["unsalvageable"], pre, post, round(st.mean(a), 2) if a else "-", round(st.mean(b), 2) if b else "-", dict(lev)))
print(f"{'lang':5}{'n':>3} fix uns  pre(m/h/o)    post(m/h/o)   OOV_m pre>post  self-rated level")
for o in sorted(out): print(f"{o[1]:5}{o[2]:>3} {o[3]:>3} {o[4]:>3}  {str(o[5]):13} {str(o[6]):13} {o[7]}>{o[8]}   {o[9]}")
print("verdicts:", dict(collections.Counter(r["proposal"].get("verdict") for r in rows)), "| cost $", round(sum((r.get("usage") or {}).get("cost") or 0 for r in rows), 2))
