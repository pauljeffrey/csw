"""Native Gemini Batch API (50% of the standard price) for the judge stages.  NEEDS GEMINI_API_KEY (Google AI Studio key) in the repo .env.  NOT TESTED YET: no Google key is in the project.

python -m csw.judging.gemini_batch submit --stage DIR/stage0_calibration --model gemini-3.8-flash [--name stage0-flash]
python -m csw.judging.gemini_batch poll   --stage DIR/stage0_calibration          # writes DIR/results/stage0_gemini-3.8-flash.jsonl in the same format run_judge_openrouter writes
The stage's gemini.jsonl (key + request lines) is uploaded with the Files API and a batch job is created from it; the job name is stored in <stage>/gemini_batch.json.
Thinking: the request files carry thinking_level 'low'; remove it from the files for models that do not take it.
"""
from __future__ import annotations

from csw.paths import env_value
import argparse
import json
import os
import re
import sys
from pathlib import Path


def client():
    from google import genai
    k = env_value("GEMINI_API_KEY") or env_value("GOOGLE_API_KEY")
    if not k:
        raise SystemExit("GEMINI_API_KEY missing in .env (create a key at https://aistudio.google.com/apikey)")
    return genai.Client(api_key=k)


def parse_json(t):
    if not t:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None


def submit(a):
    c = client(); st = Path(a.stage)
    f = c.files.upload(file=str(st / "gemini.jsonl"), config={"mime_type": "jsonl", "display_name": a.name or st.name})
    job = c.batches.create(model=f"models/{a.model}", src=f.name, config={"display_name": a.name or st.name})
    (st / "gemini_batch.json").write_text(json.dumps({"job": job.name, "model": a.model, "file": f.name}))
    print("submitted", job.name, job.state)


def poll(a):
    c = client(); st = Path(a.stage); meta = json.loads((st / "gemini_batch.json").read_text())
    job = c.batches.get(name=meta["job"]); print(job.state)
    if str(job.state).endswith("SUCCEEDED") and job.dest and job.dest.file_name:
        raw = c.files.download(file=job.dest.file_name).decode("utf-8")
        out = st.parent / "results" / f"{st.name.split('_')[0]}_{meta['model']}.jsonl"; out.parent.mkdir(exist_ok=True)
        n = bad = 0
        with open(out, "w", encoding="utf-8") as fo:
            for line in raw.split("\n"):
                if not line.strip():
                    continue
                r = json.loads(line); resp = r.get("response") or {}
                text = "".join(p.get("text", "") for p in ((resp.get("candidates") or [{}])[0].get("content") or {}).get("parts", []))
                u = resp.get("usageMetadata") or {}
                p = parse_json(text); n += 1; bad += p is None
                fo.write(json.dumps({"id": r["key"], "text": text, "parsed": p, "usage": u, "cost": None, "error": r.get("error")}, ensure_ascii=False) + "\n")
        print(f"wrote {out} ({n} rows, {bad} unparsed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("submit"); s.add_argument("--stage", required=True); s.add_argument("--model", default="gemini-3.8-flash"); s.add_argument("--name")
    p = sp.add_parser("poll"); p.add_argument("--stage", required=True)
    a = ap.parse_args()
    {"submit": submit, "poll": poll}[a.cmd](a)
