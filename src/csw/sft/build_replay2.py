"""Replay candidates from the DECONTAMINATED en->X set (the repository named by CSW_REPLAY_CLEAN_REPO; dev/devtest/test text of 22 benchmarks removed on both sides).
    python -m csw.sft.build_replay2 --out work/replay2_candidates.jsonl --stats work/replay2_stats.json [--cap 700]
One pass over data/pairs-*.parquet (each file downloaded, read, deleted). Per (iso, source_dataset) a reservoir sample of at most --cap pairs is kept together with the pair counts, so the
human-translation allow-list is chosen afterwards from the stats, not guessed. Rows: {iso, english, target, source_dataset, licence, flipped, decon_flag}."""
from __future__ import annotations
from csw.paths import env_value
import argparse, collections, json, os, random
from pathlib import Path


def tok():
    return os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--stats", required=True); ap.add_argument("--cap", type=int, default=700)
    ap.add_argument("--repo", default=env_value("CSW_REPLAY_CLEAN_REPO")); a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    api = HfApi(token=tok()); rnd = random.Random(7)
    files = sorted(f for f in api.list_repo_files(a.repo, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet"))
    seen = collections.Counter(); keep = collections.defaultdict(list); lic = {}
    for f in files:
        p = hf_hub_download(a.repo, f, repo_type="dataset", token=tok()); pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=50000, columns=["english", "target", "iso", "source_dataset", "licence", "flipped", "decon_flag"]):
            d = b.to_pydict()
            for i in range(len(d["iso"])):
                k = (d["iso"][i], d["source_dataset"][i]); seen[k] += 1; lic[k] = d["licence"][i]
                if len(d["english"][i]) < 5 or len(d["target"][i]) < 3 or len(d["english"][i]) > 1500: continue
                r = {"iso": k[0], "english": d["english"][i], "target": d["target"][i], "source_dataset": k[1], "licence": d["licence"][i], "flipped": d["flipped"][i], "decon_flag": d["decon_flag"][i]}
                v = keep[k]
                if len(v) < a.cap: v.append(r)
                else:
                    j = rnd.randrange(seen[k])
                    if j < a.cap: v[j] = r
        os.remove(os.path.realpath(p)); os.remove(p) if os.path.islink(p) else None; print("done", f, sum(seen.values()), flush=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for v in keep.values():
            for r in v: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.stats).write_text(json.dumps({f"{k[0]}|{k[1]}": {"n": n, "licence": lic[k]} for k, n in sorted(seen.items())}, indent=0))
    print("candidates", sum(len(v) for v in keep.values()), "keys", len(keep))


if __name__ == "__main__":
    main()
