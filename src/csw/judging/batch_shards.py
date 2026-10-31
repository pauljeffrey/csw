"""Gemini Tier 1 allows only 3M batch tokens active at once, so a big pool goes up in shards: a new shard is submitted whenever the quota has room (429 -> wait), finished shards are downloaded,
and at the end the shard results are merged into results_<judge>@v13.jsonl.  Resumable (state in D/shards.json).  BATCH API only.
  python -m csw.judging.batch_shards --dir work/pool_flash --judge gemini-3.8-flash [--shard 600]"""
import argparse, json, time, types
from pathlib import Path
from csw.judging import batch_judge as B
from csw.judging import judge_v11_test as J


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--judge", default="gemini-3.8-flash"); ap.add_argument("--shard", type=int, default=600)
    ap.add_argument("--lexicon", default=str(B.HERE / "work" / "lexicon.json")); ap.add_argument("--thinking", default="low"); a0 = ap.parse_args()
    d = Path(a0.dir); n = sum(1 for _ in open(d / "requests.jsonl")); st_f = d / "shards.json"
    st = json.loads(st_f.read_text()) if st_f.exists() else {"cache": None, "shards": [{"i": i, "offset": o, "job": None, "done": False} for i, o in enumerate(range(0, n, a0.shard))]}
    save = lambda: st_f.write_text(json.dumps(st))
    from google.genai import types as gt
    c = B.gclient()
    if not st["cache"]:
        system, _ = J.card_parts("v13")
        st["cache"] = c.caches.create(model=f"models/{a0.judge}", config=gt.CreateCachedContentConfig(system_instruction=system, ttl="172800s", display_name="judge-card-v13")).name; save()
    while not all(s["done"] for s in st["shards"]):
        for s in st["shards"]:
            if s["done"]: continue
            tag = f"_s{s['i']:03d}"
            if not s["job"]:
                a = types.SimpleNamespace(dir=a0.dir, judge=a0.judge, lexicon=a0.lexicon, offset=s["offset"], limit=a0.shard, thinking=a0.thinking, reasoning="low")
                lines = B.build_lines(a, st["cache"]); f = d / f"batch_{a0.judge}{tag}.jsonl"
                f.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
                try:
                    up = c.files.upload(file=str(f), config={"mime_type": "jsonl", "display_name": f.stem}); job = c.batches.create(model=f"models/{a0.judge}", src=up.name, config={"display_name": f.stem})
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e): print(time.strftime("%T"), "quota full at shard", s["i"], flush=True); break
                    raise
                s["job"] = job.name; save(); print(time.strftime("%T"), "submitted shard", s["i"], job.name, flush=True)
            else:
                job = c.batches.get(name=s["job"]); stt = str(job.state)
                if stt.endswith("SUCCEEDED"):
                    (d / f"shard{tag}.raw").write_text(c.files.download(file=job.dest.file_name).decode("utf-8")); s["done"] = True; save(); print(time.strftime("%T"), "shard", s["i"], "done", flush=True)
                elif any(k in stt for k in ("FAILED", "CANCELLED", "EXPIRED")):
                    print(time.strftime("%T"), "shard", s["i"], stt, "-> resubmit", flush=True); s["job"] = None; save()
        time.sleep(300)
    # merge: reuse poll's parsing by writing one combined raw file and a fake meta
    raw = "\n".join((d / f"shard_s{s['i']:03d}.raw").read_text() for s in st["shards"])
    (d / "combined.raw").write_text(raw)
    print("all shards done; combined raw ->", d / "combined.raw", flush=True)
    (d / f"batch_{a0.judge}.json").write_text(json.dumps({"job": "sharded", "n": n}))
    try: c.caches.delete(name=st["cache"])
    except Exception: pass


if __name__ == "__main__":
    main()
