"""Compare ablation arms on the rows they ALL generated (arm E stopped early, so the common set is E's finished rows) and pick the architecture for the full SFT by a rule fixed before the results:
    score = mean over the five tasks of the mean sentence-level chrF++ of the final text (greedy sample 0);
    winner = the SMALLEST arm whose paired-bootstrap 95% CI of (best - arm) includes 0 and whose gap to the best is under 1.0 chrF; otherwise the best arm.
python -m csw.sft.compare_arms --arms A=gens_abl/A_test.jsonl E=gens_abl/E_test.jsonl F=gens_abl/F_test.jsonl D=gens_abl/D_test.jsonl --params A=268 D=268 E=301 F=335 --out results_abl/compare"""
from __future__ import annotations
import argparse, collections, glob, json, random, statistics, sys
from pathlib import Path
from csw.sft.eval_chrf import parts


def load(spec: str) -> dict:
    d = {}
    for f in sorted(glob.glob(spec.replace(".jsonl", "*.jsonl")) if "_b" not in spec else glob.glob(spec)):
        if f.endswith(".log"): continue
        for l in open(f, encoding="utf-8"):
            try: r = json.loads(l)
            except Exception: continue
            if "generations" in r: d[r["id"]] = r
    return d


def loop(t: str) -> bool:
    w = t.split()
    if len(w) < 12: return False
    g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(g)) / len(g) < 0.6


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--params", nargs="+", default=[]); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000); a = ap.parse_args()
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2); data = {}
    for s in a.arms:
        n, p = s.split("="); data[n] = load(p)
    params = {x.split("=")[0]: float(x.split("=")[1]) for x in a.params}
    names = [n for n in data if data[n]]; ids = sorted(set.intersection(*[set(data[n]) for n in names]))
    print("arms", names, "common rows", len(ids), {n: len(data[n]) for n in names})
    task = {i: data[names[0]][i]["task"] for i in ids}; sc = {n: {} for n in names}; info = {}
    for n in names:
        rat, lp, noend, fmt = [], 0, 0, 0
        for i in ids:
            r = data[n][i]; g = r["generations"][0]; t, devs, _, has = parts(g); ref, _, _, _ = parts(r["reference"])
            sc[n][i] = chrf.sentence_score(t, [ref]).score if task[i] != "eng_m" else float("nan")
            rat.append(len(t) / max(1, len(ref))); lp += loop(t); noend += "</dev>" not in g; fmt += bool(has and len(devs) >= 2 and t)
        info[n] = {"loop_pct": 100 * lp / len(ids), "no_close_pct": 100 * noend / len(ids), "format_ok": fmt / len(ids), "len_mean": statistics.mean(rat), "len_median": statistics.median(rat)}
    tasks = sorted({t for t in task.values() if t != "eng_m"}); by_task = {t: [i for i in ids if task[i] == t] for t in tasks}
    macro = lambda n, sel: statistics.mean(statistics.mean(sc[n][i] for i in sel[t]) for t in tasks if sel[t])
    score = {n: macro(n, by_task) for n in names}; best = max(score, key=score.get); rnd = random.Random(7); diffs = {n: [] for n in names}
    for _ in range(a.boot):
        sel = {t: [rnd.choice(v) for _ in v] for t, v in by_task.items()}
        b = macro(best, sel)
        for n in names: diffs[n].append(b - macro(n, sel))
    ci = {n: (sorted(d)[int(0.025 * len(d))], sorted(d)[int(0.975 * len(d)) - 1]) for n, d in diffs.items()}
    tied = [n for n in names if n == best or (ci[n][0] <= 0 <= ci[n][1] and score[best] - score[n] < 1.0)]
    winner = min(tied, key=lambda n: (params.get(n, 0), -score[n]))
    lines = [f"common rows {len(ids)} (arms: {', '.join(names)})", f"{'arm':4}{'params':>7}{'score':>8}" + "".join(f"{t:>9}" for t in tasks) + f"{'loop%':>7}{'noclose%':>9}{'fmt_ok':>7}{'len':>6}   gap-to-best CI95"]
    for n in names:
        lines.append(f"{n:4}{params.get(n, 0):7.0f}{score[n]:8.2f}" + "".join(f"{statistics.mean(sc[n][i] for i in by_task[t]):9.2f}" for t in tasks) + f"{info[n]['loop_pct']:7.1f}{info[n]['no_close_pct']:9.1f}{info[n]['format_ok']:7.3f}{info[n]['len_mean']:6.2f}   [{ci[n][0]:+.2f}, {ci[n][1]:+.2f}]")
    lines.append(f"best {best}; tied with best: {tied}; WINNER (smallest tied) = {winner}")
    print("\n".join(lines)); Path(a.out + ".txt").write_text("\n".join(lines) + "\n")
    json.dump({"common_rows": len(ids), "score": score, "info": info, "ci_gap_to_best": ci, "best": best, "tied": tied, "winner": winner, "params_m": params}, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
