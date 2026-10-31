"""Best-of-N with the evaluator as selector: for each system and item, the greedy output and N sampled outputs are scored by the evaluator; the pick = highest evaluator score (format failures last).
Reports, per system and task, chrF++ (corpus, micro) and mean evaluator score of: greedy, a random sample, the evaluator's pick, and the oracle pick (highest chrF of the N).  No API calls.
python -m csw.rl.eval_samples --test final_v5/test_off.jsonl --evaluator runs/ev_final --system sft=gens_full/full_E_test_off.jsonl:gens_samples/sft_samples8.jsonl ... --out results_samples"""
import argparse, collections, json, random, re, sys
from pathlib import Path
import numpy as np, sacrebleu
from csw.rl import rules as R
from csw.rl.evaluator import EvaluatorScorer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", required=True); ap.add_argument("--evaluator", required=True); ap.add_argument("--system", action="append", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    sc = EvaluatorScorer(a.evaluator, device="cuda"); test = {}
    for l in open(a.test, encoding="utf-8"):
        r = json.loads(l); test[r["id"]] = r
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rnd = random.Random(0); res = {}; per_item = []
    for spec in a.system:
        name, files = spec.split("=", 1); gf, sf = files.split(":"); greedy = {}; samples = {}
        for l in open(gf, encoding="utf-8"):
            g = json.loads(l); greedy[g["id"]] = g["generations"][0]
        for l in open(sf, encoding="utf-8"):
            g = json.loads(l); samples[g["id"]] = g["samples"]
        ids = [i for i in samples if i in test and i in greedy]; cands = []; meta = []                      # candidate 0 = greedy, 1..N = samples
        for i in ids:
            t = test[i]; ref = R.parse(t["completion"] + " <end_of_turn>")["final"]
            for k, txt in enumerate([greedy[i]] + samples[i]):
                p = R.parse(txt + " <end_of_turn>"); ok = bool(p["final"]) and p["dev_ok"] if not t["task"].startswith("eng_m") else bool(p["final"])
                cands.append((t["task"], t["lang"], t.get("switch") or "", t["prompt"].split(":", 1)[1].strip(), p["final"], p["devices"], ok)); meta.append((i, k, ref, p["final"], ok))
        todo = [j for j, c in enumerate(cands) if c[6]]; ev = [-9.0] * len(cands)
        for s in range(0, len(todo), 4096):
            chunk = todo[s:s + 4096]; r = sc([(cands[j][0], cands[j][1], cands[j][2], cands[j][3], cands[j][4], cands[j][5]) for j in chunk])
            for j, v in zip(chunk, r): ev[j] = float(v)
        print(name, "scored", len(cands), "candidates", flush=True); n = len(samples[ids[0]]) + 1; by = collections.defaultdict(lambda: collections.defaultdict(list))
        for q, i in enumerate(ids):
            rows = [(k, ev[q * n + k], meta[q * n + k]) for k in range(n)]; t = test[i]
            chrf = [sacrebleu.sentence_chrf(m[3], [m[2]], word_order=2).score if m[4] else 0.0 for _, _, m in rows]
            pick = max(range(n), key=lambda k: ev[q * n + k]); orc = max(range(n), key=lambda k: chrf[k]); rs = rnd.randrange(1, n)
            for strat, k in (("greedy", 0), ("random_sample", rs), ("evaluator_pick", pick), ("oracle_chrf", orc)):
                by[t["task"]][strat].append((rows[k][2][3], rows[k][2][2], ev[q * n + k], chrf[k], rows[k][2][4]))
            per_item.append({"system": name, "id": i, "task": t["task"], "lang": t["lang"], "pick": pick, "pick_ev": ev[q * n + pick], "greedy_ev": ev[q * n], "pick_chrf": chrf[pick], "greedy_chrf": chrf[0], "ev": [round(ev[q * n + k], 3) for k in range(n)], "chrf": [round(c, 2) for c in chrf]})
        res[name] = {task: {strat: {"chrf_corpus": sacrebleu.corpus_chrf([x[0] for x in v], [[x[1] for x in v]], word_order=2).score, "evaluator": float(np.mean([x[2] for x in v if x[4]])), "fmt_fail": 1 - float(np.mean([x[4] for x in v])), "n": len(v)} for strat, v in d.items()} for task, d in by.items()}
    json.dump(res, open(out / "samples_summary.json", "w"), indent=1)
    with open(out / "samples_per_item.jsonl", "w", encoding="utf-8") as f:
        for r in per_item: f.write(json.dumps(r) + "\n")
    L = ["# Best-of-N with the evaluator as selector (corpus chrF++ per task, micro; evaluator mean over format-valid outputs)\n", "| system | task | strategy | chrF++ | evaluator | format fail |", "|---|---|---|---|---|---|"]
    for name, d in res.items():
        for task in sorted(d):
            for strat in ("greedy", "random_sample", "evaluator_pick", "oracle_chrf"): v = d[task][strat]; L.append(f"| {name} | {task} | {strat} | {v['chrf_corpus']:.2f} | {v['evaluator']:.3f} | {v['fmt_fail']:.3f} |")
    (out / "samples.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
