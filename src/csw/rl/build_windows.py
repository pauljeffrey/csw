"""Per-language length windows for the English-source tasks from the SFT references: ratio = chars(final text) / chars(English source); window = [0.95 x p5, 1.05 x p95] per (task, language) with >= 30 rows.  Catches hallucinated additions and omissions that the evaluator does not see.
python -m csw.rl.build_windows --sft final_v5/train.jsonl --out csw_rl/data/length_windows.json   (reads the SFT train references only)"""
import argparse, collections, json, re, sys
from pathlib import Path
from csw.rl import rules as R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sft", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); by = collections.defaultdict(list)
    for l in open(a.sft, encoding="utf-8"):
        if not any(t in l[:400] for t in ('"eng_m"', '"eng_hm"', '"eng_ood"')): continue
        r = json.loads(l); t = r.get("task")
        if t not in ("eng_m", "eng_hm", "eng_ood"): continue
        src = r["prompt"].split(":", 1)[-1].strip(); f = R.parse(r["completion"] + " </s>")["final"]
        if t == "eng_m": f = re.sub(r"^[a-z]{3}\s+", "", f)
        if f and src: by[f"{t}|{r['lang']}"].append(len(f) / len(src))
    q = lambda v, x: sorted(v)[min(len(v) - 1, int(x * (len(v) - 1) + 0.5))]
    out = {k: [round(0.95 * q(v, 0.05), 3), round(1.05 * q(v, 0.95), 3)] for k, v in by.items() if len(v) >= 30}
    json.dump(out, open(a.out, "w")); print(len(out), "windows;", {k: out[k] for k in list(out)[:6]})


if __name__ == "__main__":
    main()
