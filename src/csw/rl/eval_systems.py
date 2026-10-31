"""Score the outputs of several systems on the same test items with the final evaluator and the deterministic rules (no LLM calls, no reward weights): evaluator mean, format-failure rate and every rule penalty per task.
python -m csw.rl.eval_systems --test final_v5/test_off.jsonl --evaluator runs/ev_final --system sft=gens_full/full_E_test_off.jsonl --system rl_best=... --out results/systems"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
from csw.rl.reward import Reward
from csw.rl.rules import FuncWords, EN_FUNC, FR_FUNC
from csw.rl.evaluator import EvaluatorScorer
from csw.rl.lid import Lid


def ci(v, B=2000, seed=0):
    r = np.random.default_rng(seed); v = np.asarray(v, float); m = [v[r.integers(0, len(v), len(v))].mean() for _ in range(B)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", required=True); ap.add_argument("--evaluator", required=True); ap.add_argument("--system", action="append", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    here = Path(__file__).parent / "data"; fw = FuncWords(here / "funcwords.json"); fw.t.setdefault("eng", EN_FUNC); fw.t.setdefault("fra", FR_FUNC)
    rw = Reward(EvaluatorScorer(a.evaluator, device="cuda"), funcwords=fw, lid=Lid(), group=1, allowed_switch=json.load(open(here / "allowed_switch.json")), base_floors=json.load(open(here / "base_fw_floor.json")),
                lang_windows=json.load(open(here / "length_windows.json")), number_words={k: set(v) for k, v in json.load(open(here / "number_words.json")).items()})
    test = {}
    for l in open(a.test, encoding="utf-8"):
        r = json.loads(l); test[r["id"]] = r
    res = {}; per_item = []
    for spec in a.system:
        name, f = spec.split("=", 1); gens = [json.loads(l) for l in open(f, encoding="utf-8")]; items = []; comps = []; ids = []
        for g in gens:
            t = test.get(g["id"])
            if not t: continue
            sw = t.get("switch"); items.append({"task": t["task"], "lang": t["lang"], "pin": sw or "", "source": t["prompt"].split(":", 1)[1].strip(), "switch_lang": sw if sw not in (None, "None") else "eng"})
            comps.append(g["generations"][0] + " <end_of_turn>"); ids.append(g["id"])
        rew, det = rw(items, comps); res[name] = (items, rew, det, ids); print(name, len(ids), "scored", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); L = ["# Evaluator and rule scores of each system on the same test items (draft off)\n"]; summary = {}
    for name, (items, rew, det, ids) in res.items():
        by = collections.defaultdict(list)
        for it, r, d, i in zip(items, rew, det, ids):
            by[it["task"]].append((r, d)); per_item.append({"id": i, "system": name, "task": it["task"], "lang": it["lang"], "reward": r, "evaluator": d.get("evaluator"), "gate": bool(d.get("gate")), "pens": d.get("pens", {})})
        summary[name] = {}
        for task, v in by.items():
            ev = [d["evaluator"] for _, d in v if "evaluator" in d]; pens = collections.defaultdict(list)
            for _, d in v:
                for k, x in (d.get("pens") or {}).items(): pens[k].append(x)
            summary[name][task] = {"n": len(v), "reward": float(np.mean([r for r, _ in v])), "evaluator": float(np.mean(ev)), "gate_rate": float(np.mean([bool(d.get("gate")) for _, d in v])), "pens": {k: float(np.mean(x)) for k, x in pens.items()}}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    with open(out / "per_item.jsonl", "w", encoding="utf-8") as f:
        for r in per_item: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    names = list(res); tasks = sorted({t for s in summary.values() for t in s})
    L.append("| task | metric | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names))
    for t in tasks:
        for m in ("evaluator", "reward", "gate_rate"): L.append(f"| {t} | {m} | " + " | ".join(f"{summary[n][t][m]:.3f}" for n in names) + " |")
    L.append("\n## Mean rule penalty (before weighting)\n\n| task | rule | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names))
    for t in tasks:
        for k in sorted({k for n in names for k in summary[n][t]["pens"]}): L.append(f"| {t} | {k} | " + " | ".join(f"{summary[n][t]['pens'].get(k, 0):.3f}" for n in names) + " |")
    base = names[0]; idx = {(r["system"], r["id"]): r for r in per_item}; L.append(f"\n## Paired difference to {base} (mean over items, 95% bootstrap CI)\n\n| system | metric | all tasks | " + " | ".join(tasks) + " |\n|---|---|---|" + "---|" * len(tasks))
    for n in names[1:]:
        for m in ("evaluator", "reward"):
            cells = []
            for sel in [None] + tasks:
                d = [idx[(n, r["id"])][m] - r[m] for r in per_item if r["system"] == base and (n, r["id"]) in idx and (sel is None or r["task"] == sel) and r[m] is not None and idx[(n, r["id"])][m] is not None]
                lo, hi = ci(d); cells.append(f"{np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")
            L.append(f"| {n} | {m} | " + " | ".join(cells) + " |")
    (out / "systems.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
