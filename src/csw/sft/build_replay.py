"""Candidate eng->m replay pairs from HUMAN-translated data, run on the box (streams the Hub files one by one, never keeps them).
    python -m csw.sft.build_replay --out work/replay_candidates.jsonl --stats work/replay_stats.json
Sources:  (1) BeardedMonster/sft-all  sft_new_existing  task=translation, source_lang=eng  (origin in metadata; the origin is kept so it can be allow-listed afterwards),
          (2) AfriScience-MT dev references (Apache-2.0, professional),  (3) weak-language-parallel train_ready/permissive (CC0 / CC-BY; synthetic English side excluded).
Never used: NLLB / TranslatePsy / nllb_mined pairs, FineTranslations (English side is machine-generated), the benchmarks' test and devtest splits, noncommercial files.
Each row: {lang, eng, m, source, origin, licence}."""
from __future__ import annotations
import argparse, collections, json, os, re
from pathlib import Path

BANNED = re.compile(r"nllb|translatepsy|finetranslations|mined|noncommercial", re.I)


def tok():
    return os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--stats", required=True); ap.add_argument("--langs", default=""); a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    api = HfApi(token=tok()); keep = set(a.langs.split()) if a.langs else None
    st = collections.Counter(); out = open(a.out, "w", encoding="utf-8")

    def emit(lang, eng, m, source, origin, lic):
        if keep and lang not in keep: return
        if BANNED.search(f"{origin} {source} {lic}"): st["banned:" + str(origin)[:40]] += 1; return
        eng, m = " ".join(str(eng).split()), " ".join(str(m).split())
        if len(eng) < 5 or len(m) < 3 or len(eng) > 1500: return
        out.write(json.dumps({"lang": lang, "eng": eng, "m": m, "source": source, "origin": origin, "licence": lic}, ensure_ascii=False) + "\n"); st[f"{source}|{origin}|{lang}"] += 1

    rid = "BeardedMonster/sft-all"
    for f in sorted(x for x in api.list_repo_files(rid, repo_type="dataset") if "sft_new_existing" in x and x.endswith(".parquet")):
        p = hf_hub_download(rid, f, repo_type="dataset", token=tok())
        for r in pq.read_table(p, columns=["task", "source_lang", "targ_lang", "input", "response", "metadata"]).to_pylist():
            if r["task"] != "translation" or r["source_lang"] != "eng" or r["targ_lang"] == "eng" or not r["input"]: continue
            try: md = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {})
            except Exception: md = {}
            if md.get("unsure_src") or md.get("unsure_tgt"): continue
            emit(r["targ_lang"], r["input"], r["response"], "sft-all", md.get("origin") or md.get("source") or "?", md.get("licence") or "?")
        os.remove(p)
    rid = "BeardedMonster/SabiYarn-MT-benchmark-translations"
    p = hf_hub_download(rid, "eval_benchmarks/afriscience_mt_dev_text.parquet", repo_type="dataset", token=tok())
    for r in pq.read_table(p).to_pylist(): emit(r["language"], r["source_en"], r["reference"], "afriscience_mt_dev", "dsfsi/afriscience_mt", "Apache-2.0")
    rid = "BeardedMonster/sabiyarn-weak-language-parallel-2026-10-26"
    for f in sorted(x for x in api.list_repo_files(rid, repo_type="dataset") if x.startswith("train_ready/permissive/")):
        p = hf_hub_download(rid, f, repo_type="dataset", token=tok())
        for l in open(p, encoding="utf-8"):
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("english_machine_generated"): continue
            emit(r["iso"], r["english"], r["target"], "weak_parallel", r.get("source_id"), r.get("licence"))
    out.close(); Path(a.stats).write_text(json.dumps(dict(st), indent=1))
    print("done", sum(v for k, v in st.items() if not k.startswith("banned")))


if __name__ == "__main__":
    main()
