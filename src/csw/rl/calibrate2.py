"""Rate / mean of every rule penalty on references vs generations (greedy member), per task: python -m csw.rl.calibrate2 --gens FILE [--lid] --name NAME.  rows: prompt, task, lang, switch, reference, generations[0]=greedy."""
import argparse, collections, json, re, sys
from pathlib import Path
from csw.rl.reward import Reward
from csw.rl.rules import FuncWords
from csw.rl.lid import Lid


def source_of(prompt): return prompt.split(":", 1)[1].strip() if ":" in prompt else prompt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gens", required=True); ap.add_argument("--name", default="x"); ap.add_argument("--no-lid", action="store_true"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    here = Path(__file__).parent / "data"; fw = FuncWords(here / "funcwords.json"); import json as _j; rw = Reward(lambda items: [3.0] * len(items), funcwords=fw, lid=None if a.no_lid else Lid(), group=1, allowed_switch=_j.load(open(here / "allowed_switch.json")))
    rows = [json.loads(l) for l in open(a.gens, encoding="utf-8") if l.strip()]
    if a.limit: rows = rows[: a.limit]
    out = {}
    for kind in ("reference", "generation"):
        items, comps = [], []
        for r in rows:
            t = r["generations"][0] if kind == "generation" else r["reference"]
            items.append({"task": r["task"], "lang": r["lang"], "pin": "", "source": source_of(r["prompt"]), "switch_lang": r.get("switch") or "eng"}); comps.append(t + " <end_of_turn>")
        rew, det = rw(items, comps); agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for it, d in zip(items, det):
            if d.get("gate"): agg[it["task"]]["gate"].append(1.0); continue
            agg[it["task"]]["gate"].append(0.0)
            for k, v in d["pens"].items(): agg[it["task"]][k].append(v)
        out[kind] = {t: {k: (sum(v) / len(v), sum(x > 0 for x in v) / len(v)) for k, v in m.items()} for t, m in agg.items()}
    L = [f"# {a.name}: mean penalty / share >0, reference vs greedy generation\n", "| task | rule | ref mean | ref >0 | gen mean | gen >0 |", "|---|---|---|---|---|---|"]
    for t in sorted(out["reference"]):
        for k in sorted(out["reference"][t]):
            r = out["reference"][t][k]; g = out["generation"].get(t, {}).get(k, (float("nan"),) * 2); L.append(f"| {t} | {k} | {r[0]:.3f} | {100*r[1]:.0f}% | {g[0]:.3f} | {100*g[1]:.0f}% |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
