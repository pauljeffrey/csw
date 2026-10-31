"""Submit shard dirs (stage_sXX/gemini.jsonl) as the per-model enqueued-token quota allows, poll, and download each finished shard to <dir>/results/stage_sXX.raw.jsonl.  Writes <dir>/DONE when all are downloaded.
python -m csw.judging.batch_driver --dir work/stage2_both [--model gemini-3.8-flash]   (run it in the background with nohup)"""
from csw.paths import DATA_ROOT
import argparse, json, sys, time
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import gemini_batch as GB


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--model", default="gemini-3.8-flash"); a = ap.parse_args()
    d = Path(a.dir); shards = sorted(d.glob("stage_s*")); (d / "results").mkdir(exist_ok=True); log = open(d / "driver.log", "a"); c = GB.client()
    def say(m): log.write(f"{time.strftime('%H:%M:%S')} {m}\n"); log.flush()
    say(f"{len(shards)} shards")
    while True:
        done = 0
        for s in shards:
            out = d / "results" / f"{s.name}.raw.jsonl"
            if out.exists(): done += 1; continue
            f = s / "gemini_batch.json"
            if not f.exists():
                try: GB.submit(__import__("argparse").Namespace(stage=str(s), model=a.model, name=f"{d.name}-{s.name}")); say(f"submitted {s.name}")
                except Exception as e:
                    if "402" in str(e) or "prepayment credits" in str(e): say(f"{s.name}: CREDITS DEPLETED (402) -- top up AI Studio billing, then restart the driver"); (d / "problem.txt").write_text("402 prepayment credits depleted"); sys.exit(2)
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e): say(f"{s.name}: quota, retry later"); continue
                    say(f"{s.name}: submit error {str(e)[:200]}"); (d / "problem.txt").write_text(str(e)[:400]); sys.exit(1)
                continue
            job = c.batches.get(name=json.loads(f.read_text())["job"]); st = str(job.state)
            if st.endswith("SUCCEEDED"): out.write_bytes(c.files.download(file=job.dest.file_name)); say(f"{s.name} downloaded"); done += 1
            elif any(x in st for x in ("FAILED", "CANCELLED", "EXPIRED")): say(f"{s.name} {st}"); (d / "problem.txt").write_text(f"{s.name} {st}"); sys.exit(1)
        say(f"downloaded {done}/{len(shards)}")
        if done == len(shards): break
        time.sleep(240)
    (d / "DONE").touch(); say("DONE")


if __name__ == "__main__":
    main()
