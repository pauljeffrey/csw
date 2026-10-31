"""Judge a requests.jsonl through the BATCH APIs only (50% of the standard price, 24 h window). Never sends realtime calls.
  python -m csw.judging.batch_judge submit --dir D --judge gemini-3.8-flash|gpt-5.6-luna --lexicon L [--limit N]
  python -m csw.judging.batch_judge poll   --dir D --judge ...        (writes D/results_<judge>@v13.jsonl in the format judge_pilot2 report reads)
Same card (v1.3) and same per-row message as judge_pilot2 run."""
from csw.paths import DATA_ROOT
import argparse, json, os, sys, time
from pathlib import Path
HERE = DATA_ROOT
from csw.sft import devices as dv
from csw.judging import judge_v11_test as J
from csw.judging.judge_pilot2 import openai_key


STRICT = ("\n\nCALIBRATION (strict reviewer). The texts come from an automatic generator, not from experts. Grade as a strict native reviewer. 5 = no flaw a native speaker would notice anywhere in the field; "
          "4 = exactly one minor flaw; 3 = several minor flaws or one clear flaw; 2 = a major flaw (meaning, idiom, register or language wrong); 1 = unusable. Do not give 4 or 5 by default: in a typical batch about half of all grades "
          "should be 3 or lower. A 4 or 5 on an H, O or D dimension requires that you checked the exact words concerned; if you cannot verify them, return null. A switch phrase or idiom you cannot confirm as natural is weak (1), not apt (2). "
          "Always list the error spans (e) and the edits (ed) that justify every grade below 5.")


def build_lines(a, cache_name=None):
    reqs = [json.loads(l) for l in open(Path(a.dir) / "requests.jsonl", encoding="utf-8")]
    reqs = reqs[getattr(a, 'offset', 0):]
    if a.limit: reqs = reqs[: a.limit]
    lex = dv.Lexicon.load(a.lexicon); system, _ = J.card_parts("v13"); system = system + (STRICT if getattr(a, "strict", False) else ""); out = []
    for q in reqs:
        user = J.user_message(q["row"], lex, compact=True)
        if a.judge.startswith("gemini"):
            req = {"contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generation_config": {"temperature": 0, "max_output_tokens": getattr(a, "max_out", 4000), "response_mime_type": "application/json", "thinking_config": {"thinking_level": a.thinking}}}
            if cache_name: req["cached_content"] = cache_name              # the static card lives in the cache: billed at the cached-input rate
            else: req["system_instruction"] = {"parts": [{"text": system}]}
            out.append({"key": q["rid"], "request": req})
        else:
            out.append({"custom_id": q["rid"], "method": "POST", "url": "/v1/chat/completions",
                        "body": {"model": a.judge, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "response_format": {"type": "json_object"},
                                 "max_completion_tokens": 8000, "reasoning_effort": a.reasoning}})
    return out


def gclient():
    from google import genai
    return genai.Client(api_key=J.key())


def oreq(method, path, **kw):
    import requests
    r = requests.request(method, "https://api.openai.com/v1" + path, headers={"Authorization": "Bearer " + openai_key()}, timeout=600, **kw)
    r.raise_for_status(); return r


def submit(a):
    d = Path(a.dir); cache_name = None
    if a.judge.startswith("gemini") and a.cache:
        from google.genai import types
        system, _ = J.card_parts("v13")
        gc = gclient(); cache_name = gc.caches.create(model=f"models/{a.judge}", config=types.CreateCachedContentConfig(system_instruction=system, ttl="172800s", display_name="judge-card-v13")).name
        print("cache", cache_name)
    lines = build_lines(a, cache_name); f = d / f"batch_{a.judge}{a.tag}.jsonl"
    f.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    if a.judge.startswith("gemini"):
        c = gclient(); up = c.files.upload(file=str(f), config={"mime_type": "jsonl", "display_name": f.stem})
        job = c.batches.create(model=f"models/{a.judge}", src=up.name, config={"display_name": f.stem})
        meta = {"job": job.name, "state": str(job.state), "n": len(lines), "cache": cache_name}
    else:
        up = oreq("POST", "/files", data={"purpose": "batch"}, files={"file": (f.name, f.read_bytes())}).json()
        b = oreq("POST", "/batches", json={"input_file_id": up["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"}).json()
        meta = {"job": b["id"], "state": b["status"], "n": len(lines)}
    (d / f"batch_{a.judge}{a.tag}.json").write_text(json.dumps(meta)); print("submitted", a.judge, meta)


def parse(t):
    try: return json.loads(t)
    except Exception: return None


def poll(a):
    d = Path(a.dir); meta = json.loads((d / f"batch_{a.judge}{a.tag}.json").read_text()); res = d / f"results_{a.judge}@v13{a.tag}.jsonl"
    try:
        import jsonschema; _, schema = J.card_parts("v13"); val = jsonschema.Draft7Validator(schema)
    except Exception:
        val = None
    recs = []
    if a.judge.startswith("gemini"):
        if (d / "combined.raw").exists():                                  # sharded run (csw.judging.batch_shards): the shards are already downloaded and joined
            raw = (d / "combined.raw").read_text()
        else:
            c = gclient(); job = c.batches.get(name=meta["job"]); st = str(job.state); print(a.judge, st)
            if not st.endswith("SUCCEEDED"): return st
            raw = c.files.download(file=job.dest.file_name).decode("utf-8")
        for l in raw.split("\n"):
            if not l.strip(): continue
            o = json.loads(l); r = o.get("response") or {}
            try:
                txt = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"] if "text" in p and not p.get("thought")); um = r.get("usageMetadata", {})
                use = {"in": um.get("promptTokenCount", 0), "out": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0), "cached": um.get("cachedContentTokenCount", 0)}
            except Exception:
                txt, use = None, None
            recs.append((o["key"], txt, use, o.get("error")))
    else:
        b = oreq("GET", "/batches/" + meta["job"]).json(); st = b["status"]; print(a.judge, st, b.get("request_counts"))
        if st != "completed": return st
        for l in oreq("GET", "/files/" + b["output_file_id"] + "/content").text.split("\n"):
            if not l.strip(): continue
            o = json.loads(l); body = (o.get("response") or {}).get("body") or {}
            try: txt = body["choices"][0]["message"]["content"]; use = {"in": body["usage"]["prompt_tokens"], "out": body["usage"]["completion_tokens"], "cached": (body["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)}
            except Exception: txt, use = None, None
            recs.append((o["custom_id"], txt, use, o.get("error")))
    tin = tout = tc = 0
    with open(res, "w", encoding="utf-8") as fo:
        for rid, txt, use, err in recs:
            rep = parse(txt) if txt else None
            rec = {"rid": rid, "judge": a.judge + "@v13", "error": str(err)[:200] if err else None, "usage": use, "reply": rep, "schema_errors": [e.message[:80] for e in val.iter_errors(rep)] if (val and rep) else []}
            if use: tin += use["in"]; tout += use["out"]; tc += use.get("cached", 0)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pin, pout, pc = {"gpt-5.6-luna": (0.10, 0.60, 0.01)}.get(a.judge, (0.375, 1.875, 0.0375))
    cost = (tin - tc) * pin / 1e6 + tc * pc / 1e6 + tout * pout / 1e6
    if meta.get("cache"):
        try:
            gc = gclient(); gc.caches.delete(name=meta["cache"])
        except Exception: pass
    print(f"wrote {len(recs)} results -> {res}; tokens in {tin} (cached {tc}) out {tout}; BATCH cost ${cost:.4f} (${cost / max(1, len(recs)):.5f}/row)"); return "done"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["submit", "poll"]); ap.add_argument("--dir", required=True); ap.add_argument("--judge", required=True)
    ap.add_argument("--lexicon", default=str(HERE / "work" / "lexicon.json")); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", action="store_true", help="Gemini: put the static card in an explicit context cache"); ap.add_argument("--tag", default="", help="suffix for the batch/result files, e.g. _cache")
    ap.add_argument("--strict", action="store_true", help="append the strict-reviewer calibration paragraph to the card"); ap.add_argument("--max-out", type=int, default=4000, help="Gemini max_output_tokens per request (thinking included)"); ap.add_argument("--thinking", default="low"); ap.add_argument("--reasoning", default="low"); a = ap.parse_args()
    {"submit": submit, "poll": poll}[a.cmd](a)
