"""Merge every Gemini audit result into ONE reviewable file (no API calls).  python -m csw.datagen.build_audit_review  ->  exemplars/gold_audit_review.jsonl
Per gold row: original fields, proposed fields (null when no usable correction -> original is kept), scores, verdict, language proficiency, review points,
and local checks (gates on the proposal; share of m_text words never seen in real corpus text before/after)."""
from csw.paths import DATA_ROOT
import json, sys, collections
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import cs_prompts as pr
from csw.datagen import gates
from csw.datagen import seeds as sm
from csw.datagen.profiles import P
E = HERE / "exemplars"
def read(n):
    p = E / n
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()] if p.exists() else []
K = lambda r: (r["lang"], r["eng"], r["m_text"], r["hm"], r["ood"])
FIELDS = ("eng", "m_text", "hm", "ood")
usable = lambda p: p.get("verdict") not in (None, "error") and all(isinstance(p.get(k), str) and p[k].strip() for k in FIELDS)
full = lambda p: isinstance(p.get("scores_before"), dict) and isinstance(p.get("scores_after"), dict) and isinstance(p.get("language_proficiency"), dict)
prop = {}                                              # key -> proposal record (best: complete > usable)
for n in ("gold_audit_flagged", "gold_audit_rest", "gold_audit_retry", "gold_audit.pilot"):
    for r in read(f"{n}.gemini.jsonl"):
        p = r["proposal"]
        if not usable(p): continue
        if K(r) not in prop or (full(p) and not full(prop[K(r)]["proposal"])): prop[K(r)] = r
scores = {(r["lang"], r["eng"], r["m_text"]): r["score"] for r in read("gold_audit_scores.gemini.jsonl")}
a = pr.load_aug_gold(); gold = [r for v in a.values() for r in v] + [g for g in pr.load_gold() if "status" not in g]
flags = {(r["lang"], r["eng"]): r["flag_reason"] for r in read("review_round2.jsonl")}
vocab = {}
def voc(l):
    if l not in vocab: vocab[l] = {w for s in sm.load(l) for w in gates.words(s["text"])}
    return vocab[l]
def oov(t, l):
    ws = [w for w in gates.words(t) if w not in gates.EN]; v = voc(l)
    return round(sum(w not in v for w in ws) / len(ws), 3) if v and ws else None
out = []
for g in gold:
    r = prop.get(K(g)); p = r["proposal"] if r else None
    row = {"lang": g["lang"], "genre": g["genre"], "switch": g["switch"], "dialect": g.get("dialect"), "source_id": g.get("source_id"),
           "flag_reason": flags.get((g["lang"], g["eng"]), ""), "original": {k: g[k] for k in FIELDS}, "proposed": None, "use": "original", "audit_source": None}
    if p:
        row["proposed"] = {k: p[k] for k in FIELDS}
        if full(p):
            row.update(audit_source="realtime", verdict=p["verdict"], scores_original=p["scores_before"], scores_proposed=p["scores_after"], level=p["language_proficiency"].get("level"),
                       can_write_reliably=p["language_proficiency"].get("can_write_reliably"), weak_areas=p["language_proficiency"].get("weak_areas"), review_points=p.get("notes"), changes=p.get("changes"))
        else:
            s = scores.get((g["lang"], g["eng"], g["m_text"])) or {}
            row.update(audit_source="batch+scoring" if s else "batch(no scores)", verdict=s.get("verdict") or p.get("verdict"), scores_original=s.get("scores_original"), scores_proposed=s.get("scores_correction"),
                       level=s.get("level"), can_write_reliably=s.get("can_write_reliably"), weak_areas=s.get("weak_areas"), review_points=s.get("review_points") or p.get("notes"), changes=p.get("changes"))
        v = str(row["verdict"])
        row["use"] = "original" if v in ("unsalvageable", "keep_original") else "proposed"
        d = dict(g, **row["proposed"]); d["scenario"] = {"switch": g["switch"]}; gt = gates.check(d, P[g["lang"]])
        row["checks"] = {"gate_fail": gt["fail"], "gate_warn": gt["warn"], "oov_m_original": oov(g["m_text"], g["lang"]), "oov_m_proposed": oov(p["m_text"], g["lang"]),
                         "hm_english_fn_share_original": round(gates.ratio(gates.words(g["hm"]), gates.EN), 2), "hm_english_fn_share_proposed": round(gates.ratio(gates.words(p["hm"]), gates.EN), 2)}
    else:
        row["verdict"] = "no_audit"                      # credit ran out before this row got a usable correction -> original kept
    row["feedback"] = ""
    out.append(row)
(E / "gold_audit_review.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
C = collections.Counter
print(len(out), "gold rows ->", "exemplars/gold_audit_review.jsonl")
print("audit source:", dict(C(r["audit_source"] for r in out))); print("use:", dict(C(r["use"] for r in out))); print("verdict:", dict(C(str(r.get("verdict")) for r in out)))
