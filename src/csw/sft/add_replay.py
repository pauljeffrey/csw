"""Add the enlarged replay (work/replay3.jsonl) to an SFT build without touching val / test: same split rule and English-leak guard as build_sft2.
python -m csw.sft.add_replay --base work/final_v4/C --eval-ids work/csw_sft_v3/eval_ids.json --rows work/post_rows --replay work/replay3.jsonl --out work/final_v5"""
import argparse, json, shutil, sys
from pathlib import Path
from csw.sft import fmt
from csw.sft import build_sft as B
from csw.sft import fmt2
from csw.sft.build_sft2 import eng_sha


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--eval-ids", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--replay", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--test-files", nargs="*", default=[], help="held-out files whose English prompts must not appear in training (eng_* tasks)"); a = ap.parse_args()
    ev = set(json.load(open(a.eval_ids))); ev = set(ev) if not isinstance(ev, dict) else set(ev.keys()); shas = set()
    for f in Path(a.rows).glob("*.raw.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in ev: shas.add(eng_sha(r["eng"]))
    for tf in a.test_files:
        for l in open(tf, encoding="utf-8"):
            e = json.loads(l)
            if e["task"].startswith("eng_"): shas.add(eng_sha(B.clean(e["prompt"].split(":  ", 1)[-1]))); shas.add(eng_sha(e["prompt"].split(":  ", 1)[-1]))
    print("English sentences guarded:", len(shas))
    base = [json.loads(l) for l in open(Path(a.base) / "train.jsonl", encoding="utf-8")]; have = {(e["prompt"], e["completion"]) for e in base}; ids = {e["id"] for e in base}
    new, st = [], {"added": 0, "dup": 0, "val_or_test_split": 0, "unknown_lang": 0}
    for l in open(a.replay, encoding="utf-8"):
        x = json.loads(l)
        if x["lang"] not in fmt.NAMES: st["unknown_lang"] += 1; continue
        if eng_sha(x["eng"]) in shas or B.split_of(x["eng"], "x") != "train": st["val_or_test_split"] += 1; continue
        e = {"id": f"replay-{eng_sha(x['eng'])[:10]}-{x['lang']}", "lang": x["lang"], "task": "eng_m", "split": "train", "source": x.get("source", "human"),
             "prompt": fmt2.prompt("eng_m", x["lang"], None, B.clean(x["eng"])), "completion": fmt.completion_m_text(x["lang"], B.clean(x["m"])), "switch": None, "meta": {}}
        if (e["prompt"], e["completion"]) in have or e["id"] in ids: st["dup"] += 1; continue
        have.add((e["prompt"], e["completion"])); new.append(e); st["added"] += 1
    out = Path(a.out) / "C"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for e in base + new: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    for n in ("val.jsonl", "test_on.jsonl", "test_off.jsonl"): shutil.copy(Path(a.base) / n, out / n)
    n_m = sum(e["task"] == "eng_m" for e in base) + len(new)
    print(st, f"train {len(base)} -> {len(base) + len(new)}; eng_m {sum(e['task']=='eng_m' for e in base)} -> {n_m} = {100*n_m/(len(base)+len(new)):.1f}% of train")


if __name__ == "__main__":
    main()
