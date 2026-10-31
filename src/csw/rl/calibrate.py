"""Penalty distributions of the deterministic rules on real full_E generations (and on the references), plus data-derived length windows.
python -m csw.rl.calibrate --gens work/rl_calib/full_E_test_off.jsonl --out work/rl_calib/report"""
import argparse, json, re, statistics, sys, collections
from pathlib import Path
from csw.rl import rules as R

SRC = re.compile(r"^(?:<[a-z]+>\s*)+(?:<[a-z]{3}>(?:\+<[a-z]{3}>)?:|<[a-z]{3}>:)?\s*")


def source_of(prompt): return prompt.split(":", 1)[1].strip() if ":" in prompt else prompt


def pct(v, q): v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gens", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.gens, encoding="utf-8") if l.strip()]; by = collections.defaultdict(lambda: collections.defaultdict(list)); win = collections.defaultdict(list)
    for r in rows:
        task = r["task"]
        if task == "eng_m": continue
        src = source_of(r["prompt"]); ref = R.parse(r["reference"] + " </s>")
        if ref["final"]: win[task].append(len(ref["final"]) / max(1, len(src)))
        for gi, g in enumerate(r["generations"][:2]):
            p = R.parse(g + " </s>"); tgt = task.split("_")[1]; d = by[task]
            d["n"].append(1); d["format_fail"].append(R.format_penalty(p, task)[0])
            if not p["final"]: continue
            d["len_ratio"].append(len(p["final"]) / max(1, len(src))); d["loop"].append(R.loop_penalty(p["final"])[0]); d["number"].append(R.number_penalty(p["final"], src)[0])
            d["english"].append(R.english_leak(p["final"], tgt == "hm")[0]); d["draft_ratio"].append(len(p["draft"]) / max(1, len(src)) if p["draft"] else 0.0)
            if tgt == "hm": d["switch"].append(R.switch_penalty(p["final"], p["devices"])[0]); d["switch_share"].append(R.switch_penalty(p["final"], p["devices"])[1]["switch_share"])
    L = ["# Rule penalties on full_E generations (greedy + 1 sample, test set, prompts WITHOUT eos in the stored text, so the eos rule is not measured here)\n", "Reference length ratio (final text / source chars), 5th / 50th / 95th percentile -> proposed windows:\n"]
    prop = {}
    for t, v in sorted(win.items()):
        lo, md, hi = pct(v, .05), pct(v, .5), pct(v, .95); prop[t] = [round(lo, 2), round(hi, 2)]; L.append(f"- {t}: {lo:.2f} / {md:.2f} / {hi:.2f}  (n={len(v)})")
    L.append("\n| task | n | format fail | len ratio p5/p50/p95 | outside proposed window | loop>0 | number>0 | english>0 | switch pen mean | switch share p5/p50/p95 | draft present |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for t, d in sorted(by.items()):
        n = len(d["n"]); lr = d["len_ratio"]; lo, hi = prop.get(t, (0.6, 1.8)); out = sum(x < lo or x > hi for x in lr) / max(1, len(lr)); f = lambda k: f"{100 * sum(x > 0 for x in d[k]) / max(1, len(d[k])):.1f}%"
        L.append(f"| {t} | {n} | {100 * sum(d['format_fail']) / n:.1f}% | {pct(lr, .05):.2f}/{pct(lr, .5):.2f}/{pct(lr, .95):.2f} | {100 * out:.1f}% | {f('loop')} | {f('number')} | {f('english')} | {statistics.mean(d['switch']) if d['switch'] else float('nan'):.2f} | " + (f"{pct(d['switch_share'], .05):.2f}/{pct(d['switch_share'], .5):.2f}/{pct(d['switch_share'], .95):.2f}" if d["switch_share"] else "-") + f" | {100 * sum(x > 0 for x in d['draft_ratio']) / max(1, len(d['draft_ratio'])):.1f}% |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out + ".md").write_text("\n".join(L)); json.dump(prop, open(a.out + "_windows.json", "w")); print("\n".join(L))


if __name__ == "__main__":
    main()
