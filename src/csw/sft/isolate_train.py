"""Remove from train every example that shares a word 8-gram (source text or final text) with any TEST example, so test is isolated from train by text, not only by row.
    python -m csw.sft.isolate_train --sft work/csw_sft_v3 [--out work/csw_sft_v3i]
Writes <out>/train.jsonl (filtered), copies val/test/test_rand/stats, and isolation_report.json with the removed counts per task and language. Test rows are never touched."""
import argparse, collections, json, re, shutil
from pathlib import Path
from csw.sft.audit_split import final, grams, src


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sft", required=True); ap.add_argument("--out", default=""); a = ap.parse_args()
    d = Path(a.sft); o = Path(a.out or str(d) + "i"); o.mkdir(parents=True, exist_ok=True)
    tg = set()
    for l in open(d / "test.jsonl", encoding="utf-8"):
        e = json.loads(l); tg |= grams(src(e["prompt"]))
        if e["task"] != "eng_m": tg |= grams(final(e["completion"]))
    print("test 8-grams", len(tg)); kept = dropped = 0; by = collections.Counter(); byl = collections.Counter()
    with open(d / "train.jsonl", encoding="utf-8") as fi, open(o / "train.jsonl", "w", encoding="utf-8") as fo:
        for l in fi:
            e = json.loads(l); hit = bool(grams(src(e["prompt"])) & tg) or (e["task"] != "eng_m" and bool(grams(final(e["completion"])) & tg))
            if hit: dropped += 1; by[e["task"]] += 1; byl[e["lang"]] += 1
            else: kept += 1; fo.write(l)
    for n in ("val.jsonl", "test.jsonl", "test_rand.jsonl", "stats.json", "eval_ids.json"):
        if (d / n).exists(): shutil.copy(d / n, o / n)
    rep = {"kept": kept, "dropped": dropped, "dropped_by_task": dict(by), "dropped_by_lang": dict(byl.most_common())}
    (o / "isolation_report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("kept", "dropped", "dropped_by_task")}), "top langs", dict(byl.most_common(6)))


if __name__ == "__main__":
    main()
