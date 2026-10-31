"""Small live test of the rubric v1.1 judge card against the Gemini API (direct, GEMINI_API_KEY from .env).

python -m csw.judging.judge_v11_test --pool DIR/judge_pool.jsonl --rows DIR/out_main --langs yor tum --per-tier 1 --out results.jsonl [--model gemini-3.8-flash]

For each language it takes the first eval row of each proxy tier (good / medium / bad), builds the per-row user message the card expects, calls the model, validates the reply against the
card's JSON Schema and prints a readable summary.  Used before any batch is built, to see whether the output is what we want.
"""
from __future__ import annotations

from csw.paths import DATA_ROOT, PAPER_DIR, env_value
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = DATA_ROOT
from csw.datagen import gates                                   # noqa: E402
from csw.datagen.profiles import P                         # noqa: E402
from csw.sft import devices as dv               # noqa: E402

CARD_FILE = PAPER_DIR / "judge_prompt_v1.txt"


def card_parts(card: str = "v12") -> tuple[str, dict]:
    if card == "v13":                                   # compact card: instructions only; the schema is kept in a separate file for offline validation
        return (PAPER_DIR / "judge_prompt_v1.3.txt").read_text(encoding="utf-8").strip(), json.loads((PAPER_DIR / "judge_schema_v1.3.json").read_text(encoding="utf-8"))
    t = CARD_FILE.read_text(encoding="utf-8")
    head = t.split("=== JSON SCHEMA", 1)[0].split("\n", 1)[1].strip()            # the static card without its header line
    schema_txt = t.split("=== JSON SCHEMA (draft-07) ===", 1)[1].split("=== EXAMPLE", 1)[0].strip()
    example = t.split("=== EXAMPLE (validates against the schema) ===", 1)[1].split("=== NOTES", 1)[0].strip()
    system = head + "\n\nJSON SCHEMA:\n" + schema_txt + "\n\nEXAMPLE OF A VALID REPLY (values are illustrative only):\n" + example
    return system, json.loads(schema_txt)


def digits(t: str) -> list[str]:
    return sorted(set(re.findall(r"\d+(?:[.,]\d+)?", t)))


def user_message(r: dict, lex: dv.Lexicon, compact: bool = False) -> str:
    p = P[r["lang"]]; sc = r.get("scenario") or {}
    h = dv.hm_devices(r, lex)
    plan = h["devices"]
    idioms = [d for d in (r.get("devices") or []) if isinstance(d, str)][:8]
    eng_digits = digits(r["eng"])
    ew = set(gates.words(r["eng"]))

    def copy_share(t):
        w = gates.words(t)
        return round(sum(x in ew and len(x) >= 4 for x in w) / max(1, len(w)), 2)

    rules = {"words": {k: len(r[k].split()) for k in ("eng", "m_text", "hm", "ood")},
             "digits_in_eng": eng_digits, "digits_missing": {k: [d for d in eng_digits if d not in r[k]] for k in ("m_text", "hm", "ood")},
             "hm_switch_share": h["switch_ratio"], "hm_equals_m": r["hm"].strip() == r["m_text"].strip(), "ood_equals_m": r["ood"].strip() == r["m_text"].strip(),
             "eng_word_share": {k: copy_share(r[k]) for k in ("m_text", "hm", "ood")},
             "plan_found_in_hm": sum(x.lower() in r["hm"].lower() for x in plan) / max(1, len(plan)) if plan else None}
    msg = {"id": r["id"], "language": {"name": p.name, "code": p.code}, "dialect": sc.get("dialect") or "standard", "genre": sc.get("genre"), "register": sc.get("register"),
           "pin": sc.get("switch"), "profile": "unknown", "eng": r["eng"], "m": r["m_text"], "h": r["hm"], "o": r["ood"],
           "P": {f"P{i + 1}": x for i, x in enumerate(plan)}, "I": {f"I{i + 1}": x for i, x in enumerate(idioms)}, "units": "not tagged in this test", "rules": rules}
    if compact:                                          # drop constants and numbers the judge can see itself
        for k in ("profile", "units", "register"):
            msg.pop(k, None) if (k != "register" or not msg.get(k)) else None
        ru = {"digits_missing": {k: v for k, v in rules["digits_missing"].items() if v}, "hm_switch_share": rules["hm_switch_share"], "eng_word_share": rules["eng_word_share"], "plan_found_in_hm": rules["plan_found_in_hm"]}
        if rules["hm_equals_m"] or rules["ood_equals_m"]:
            ru["hm_equals_m"], ru["ood_equals_m"] = rules["hm_equals_m"], rules["ood_equals_m"]
        if not ru["digits_missing"]:
            ru.pop("digits_missing")
        msg["rules"] = ru; msg.pop("id", None)
    return json.dumps(msg, ensure_ascii=False)


def call(model: str, system: str, user: str, key: str, thinking: str = "low", max_out: int = 4000) -> dict:
    body = {"system_instruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_out, "responseMimeType": "application/json"}}
    if thinking:
        body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", json.dumps(body).encode(),
                                 {"x-goog-api-key": key, "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:400]}"}
    out["_sec"] = round(time.time() - t0, 1)
    return out


def key() -> str:
    k = env_value("GEMINI_API_KEY")
    if not k:
        raise SystemExit("GEMINI_API_KEY missing in .env")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--per-tier", type=int, default=1); ap.add_argument("--out", required=True); ap.add_argument("--model", default="gemini-3.8-flash"); ap.add_argument("--lexicon")
    ap.add_argument("--thinking", default="low")
    a = ap.parse_args()
    pool = [json.loads(l) for l in open(a.pool, encoding="utf-8") if l.strip()]
    want = {}
    for r in pool:
        if r["lang"] in a.langs and r["split"] == "eval":
            n = sum(1 for v in want.values() if v["lang"] == r["lang"] and v["tier"] == r["tier"])
            if n < a.per_tier:
                want[r["id"]] = r
    rows = {}
    for lang in a.langs:
        for line in open(Path(a.rows) / f"{lang}.raw.jsonl", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["id"] in want:
                    rows[r["id"]] = r
    lex = dv.Lexicon.load(a.lexicon)
    system, schema = card_parts(); k = key()
    try:
        import jsonschema
    except ImportError:
        jsonschema = None
    outf = open(a.out, "w", encoding="utf-8"); tot_in = tot_out = 0
    for rid, meta in want.items():
        r = rows.get(rid)
        if not r:
            continue
        user = user_message(r, lex)
        resp = call(a.model, system, user, k, a.thinking)
        if "error" in resp:
            print(rid, resp["error"]); continue
        text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"] if "text" in p)
        um = resp.get("usageMetadata", {}); tot_in += um.get("promptTokenCount", 0); tot_out += um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
        try:
            parsed = json.loads(text); ok_json = True
        except Exception:
            parsed, ok_json = None, False
        errs = []
        if ok_json and jsonschema:
            v = jsonschema.Draft7Validator(schema); errs = [f"{'/'.join(map(str, e.absolute_path))}: {e.message[:80]}" for e in v.iter_errors(parsed)]
        rec = {"id": rid, "lang": r["lang"], "tier": meta["tier"], "proxy": meta["proxy"], "usage": um, "sec": resp.get("_sec"), "json_ok": ok_json, "schema_errors": errs, "reply": parsed if ok_json else text, "user_message": json.loads(user)}
        outf.write(json.dumps(rec, ensure_ascii=False) + "\n"); outf.flush()
        print(f"== {r['lang']} {meta['tier']} {rid} | tokens in {um.get('promptTokenCount')} out {um.get('candidatesTokenCount')} thoughts {um.get('thoughtsTokenCount')} | {resp.get('_sec')}s | json {'ok' if ok_json else 'BAD'} schema errors {len(errs)} {errs[:3]}")
    print(f"total tokens in {tot_in} out(incl thoughts) {tot_out}; batch-price estimate ${tot_in * 0.375e-6 + tot_out * 1.875e-6:.4f}, realtime ${tot_in * 0.75e-6 + tot_out * 3.75e-6:.4f}")


if __name__ == "__main__":
    main()
