"""Live A/B script (not a unit test): A/B of the old vs new (PIN RULE) generation prompt on identical task cards, Gemma-4-31B-it via the Gemini API, scored by gates.py.
python tests/test_pin_prompt.py --langs zul swh kin hau amh som orm sot --per-lang 6"""
import pytest

pytest.skip("live A/B against a remote model API; needs an API key and the historical prompt snapshot cs_prompts.before_pinrule.py, which is not part of this repository", allow_module_level=True)

import argparse, json, sys, threading, urllib.request, time, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from csw.paths import DATA_ROOT, env_value
HERE = DATA_ROOT
from csw.datagen import gates
from csw.datagen import cs_prompts as prompts
from csw.datagen import cs_pipeline as cp
from csw.datagen.profiles import P
from csw.judging import judge_v11_test as J

OLD = None
import importlib.util
spec = importlib.util.spec_from_file_location("old_prompts", HERE / "cs_prompts.before_pinrule.py"); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
NEW_TEXT, OLD_TEXT = prompts.STATIC_SYSTEM, old.STATIC_SYSTEM

def okey():
    return env_value("OPENROUTER_API_KEY")


def call(model, messages, key, max_out):
    sysm = "\n\n".join(m["content"] for m in messages if m["role"] == "system"); usr = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    body = {"system_instruction": {"parts": [{"text": sysm}]}, "contents": [{"role": "user", "parts": [{"text": usr}]}],
            "generationConfig": {"temperature": 0.6, "topP": 0.95, "topK": 40, "maxOutputTokens": max_out + 6000, "responseMimeType": "application/json"}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", json.dumps(body).encode(), {"x-goog-api-key": key, "Content-Type": "application/json"})
    for t in range(2):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=900) as r:
                o = json.loads(r.read())
            print("ok", round(time.time() - t0), "s", o.get("usageMetadata", {}).get("candidatesTokenCount"), o.get("usageMetadata", {}).get("thoughtsTokenCount"), flush=True)
            return "".join(p.get("text", "") for p in o["candidates"][0]["content"]["parts"] if "text" in p)
        except Exception as e:
            print("call error", str(e)[:150], getattr(e, "read", lambda: b"")()[:200], flush=True); time.sleep(5)
    return None


def extract(txt):
    dec = json.JSONDecoder(); best = None
    for i, ch in enumerate(txt or ""):
        if ch == "{":
            try:
                o, _ = dec.raw_decode(txt[i:])
            except Exception:
                continue
            if isinstance(o, dict) and "m_text" in o and (best is None or len(json.dumps(o)) > len(json.dumps(best))):
                best = o
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--langs", nargs="+", required=True); ap.add_argument("--per-lang", type=int, default=6); ap.add_argument("--model", default="gemma-4-31b-it"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); key = J.key(); gold = prompts.load_gold(); jobs = []
    for lang in a.langs:
        cards = [c for c in cp.make_plan(lang, 400) if c["switch"] not in ("English",) and "Pidgin" not in c["switch"] and "Sheng" not in c["switch"]][: a.per_lang]
        for c in cards:
            for variant in ("old", "new"):
                jobs.append((lang, c, variant))
    print(len(jobs), "calls", flush=True); res = []; lock = threading.Lock()
    def one(j):
        lang, c, variant = j
        prompts.STATIC_SYSTEM = OLD_TEXT if variant == "old" else NEW_TEXT
        msgs = prompts.gen_prompt(P[lang], c, gold)
        txt = call(a.model, msgs, key, cp.LENGTHS[c["length"]][2])
        d = extract(txt)
        rec = {"lang": lang, "variant": variant, "id": c["id"], "switch": c["switch"], "ok": bool(d)}
        if d and all(d.get(k) for k in ("eng", "m_text", "hm", "ood")):
            row = {"lang": lang, "scenario": {"switch": c["switch"], "length_spec": "v2", "n_sent": c["n_sent"]}, **{k: d[k] for k in ("eng", "m_text", "hm", "ood")}}
            lk = gates.en_leak(row); rec.update({"leak": lk, "hm": d["hm"], "eng": d["eng"], "m_text": d["m_text"]})
        with lock:
            res.append(rec)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, jobs))
    json.dump(res, open(a.out, "w"), ensure_ascii=False)
    for v in ("old", "new"):
        r = [x for x in res if x["variant"] == v]; g = [x for x in r if x.get("leak")]
        sh = [x["leak"][1] for x in g]
        print(v, "calls", len(r), "parsed", sum(x["ok"] for x in r), "scored", len(g), "mean leak share %.3f median %.3f" % (st.mean(sh), st.median(sh)), "fail rate (>=2 words & >6%%): %.2f" % (sum(x["leak"][0] >= 2 and x["leak"][1] > gates.EN_LEAK_MAX_SHARE for x in g) / len(g)), "clean (0 words): %.2f" % (sum(x["leak"][0] == 0 for x in g) / len(g)))

if __name__ == "__main__":
    main()
