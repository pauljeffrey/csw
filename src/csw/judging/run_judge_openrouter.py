"""Run a judge-batch stage through OpenRouter (the only Gemini access in the project .env; there is no Google key for the native Batch API).

python -m csw.judging.run_judge_openrouter --stage DIR/stage0_calibration --model google/gemini-3.8-flash:batch --out DIR/results/stage0_gemini-3.8-flash.jsonl [--limit N] [--conc 8] [--effort low]

Reads <stage>/openai.jsonl (system + user messages), resumes from --out, writes {id, text, parsed, usage, cost}.  OpenRouter lists a ':batch' variant at the Batch price; the plain id is realtime
(about 2.2x).  Spend is printed as it goes; --max-usd stops the run when the estimate passes it.
"""
from __future__ import annotations

from csw.paths import env_value
import argparse
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"


def key() -> str:
    k = env_value("OPENROUTER_API_KEY")
    if not k:
        raise SystemExit("OPENROUTER_API_KEY missing")
    return k


def parse_json(t: str):
    if not t:
        return None
    t = t.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def call(k: str, model: str, messages: list, max_tokens: int, effort: str | None, retries: int = 4) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "response_format": {"type": "json_object"}, "usage": {"include": True}}
    if effort:
        body["reasoning"] = {"effort": effort}
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), {"Authorization": f"Bearer {k}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except Exception as e:                      # rate limits and transient errors
            err = str(e)[:200]
            time.sleep(3 * (i + 1))
    return {"error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--conc", type=int, default=8); ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=2500); ap.add_argument("--max-usd", type=float, default=0.0)
    a = ap.parse_args()
    reqs = [json.loads(l) for l in open(Path(a.stage) / "openai.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        reqs = reqs[: a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}
    todo = [r for r in reqs if r["custom_id"] not in done]
    k = key(); lock = threading.Lock(); tot = {"cost": 0.0, "n": 0, "bad": 0}
    fo = open(out, "a", encoding="utf-8")

    def one(r):
        if a.max_usd and tot["cost"] > a.max_usd:
            return
        resp = call(k, a.model, r["body"]["messages"], a.max_tokens, a.effort or None)
        text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") if "choices" in resp else None
        u = resp.get("usage") or {}
        rec = {"id": r["custom_id"], "text": text, "parsed": parse_json(text), "usage": u, "cost": u.get("cost"), "error": resp.get("error")}
        with lock:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
            tot["n"] += 1; tot["cost"] += float(u.get("cost") or 0); tot["bad"] += rec["parsed"] is None
            if tot["n"] % 25 == 0:
                print(f"{tot['n']}/{len(todo)} done, bad={tot['bad']}, spend ~${tot['cost']:.3f}", flush=True)

    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(one, todo))
    print(json.dumps({"done": tot["n"], "unparsed": tot["bad"], "spend_usd": round(tot["cost"], 4)}))


if __name__ == "__main__":
    main()
