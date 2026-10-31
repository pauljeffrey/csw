"""Analyse the Flash batch results: (1) the standard v2 prompt on the stratified items, (2) the old strict prompt on the 300-item subset, (3) agreement between the two prompts.
python -m csw.judging.analyze_judges --results GLOB[,GLOB] --meta META.json --out DIR"""
import argparse, collections, glob, json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from csw.judging import compact_judge as C

W = Path("work/judge_v2")
SYS = ["sft", "rl_best", "rl_last", "reference"]; TASKS = ["eng_hm", "m_hm", "eng_ood", "m_ood", "hm_ood"]; LAB = {"sft": "SFT", "rl_best": "RL best", "rl_last": "RL latest", "reference": "generator reference"}


def texts(path):
    for f in sorted(glob.glob(str(path))):
        for l in open(f, encoding="utf-8"):
            o = json.loads(l); resp = o.get("response") or {}
            try:
                txt = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"] if "text" in p and not p.get("thought")); yield o["key"], json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()), resp.get("usageMetadata") or {}
            except Exception: yield o.get("key"), None, resp.get("usageMetadata") or {}


def ci(d, B=2000):
    r = np.random.default_rng(0); d = np.asarray(d, float); m = [d[r.integers(0, len(d), len(d))].mean() for _ in range(B)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True, help="comma-separated globs of raw batch result files"); ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True); ap.add_argument("--title", default="judge"); ap.add_argument("--price", default="0.375,1.875"); ap.add_argument("--old", default=""); args = ap.parse_args()
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True); pin, pout = map(float, args.price.split(","))
    meta = {m["key"]: m for m in json.load(open(args.meta))}; rows = {}; bad = 0; tin = tout = 0
    for key, rep, um in (x for g in args.results.split(",") for x in texts(g)):
        tin += um.get("promptTokenCount", 0); tout += um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
        if rep is None or not isinstance(rep.get("scores"), dict): bad += 1; continue
        rows[key] = rep["scores"]
    L = [f"# {args.title}, standard v2 prompt: {len(rows)} graded of {len(meta)} requests (unparseable {bad}); tokens in {tin:,} out {tout:,}; batch cost ${(tin * pin + tout * pout) / 1e6:.2f}\n"]
    crit = {"hm": ["meaning", "base_language", "switch_language", "language_compliance", "switching_naturalness", "overall"], "ood": ["meaning", "language_correctness", "idiom_use", "naturalness", "language_compliance", "overall"]}
    ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 1 <= v <= 5
    per = collections.defaultdict(dict)                                                       # (task, item) -> system -> mean of the valid criteria, plus overall
    for k, sc in rows.items():
        if k not in meta: continue
        m = meta[k]; item = k.rsplit("|", 1)[0]; kind = "hm" if m["task"].endswith("hm") else "ood"; v = [sc.get(c) for c in crit[kind] if ok(sc.get(c))]
        per[(m["task"], item)][m["system"]] = {"mean": float(np.mean(v)) if v else None, "overall": float(sc["overall"]) if ok(sc.get("overall")) else None, "sc": sc}
    L += ["## Mean criterion average (1-5) per system and task, and paired difference to SFT (95% bootstrap CI over items)\n", "| task | " + " | ".join(LAB[s] for s in SYS) + " | RL best - SFT | RL latest - SFT | reference - SFT |", "|---|" + "---|" * 7]
    res = {}
    for t in TASKS + ["all"]:
        items = [v for (tt, _), v in per.items() if t == "all" or tt == t]; cells = []
        for s in SYS: x = [v[s]["mean"] for v in items if s in v and v[s]["mean"] is not None]; cells.append(f"{np.mean(x):.3f}" if x else "-")
        diffs = []
        for s in ("rl_best", "rl_last", "reference"):
            d = [v[s]["mean"] - v["sft"]["mean"] for v in items if s in v and "sft" in v and v[s]["mean"] is not None and v["sft"]["mean"] is not None]
            if d: lo, hi = ci(d); diffs.append(f"{np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]"); res[(t, s)] = (float(np.mean(d)), lo, hi, len(d))
            else: diffs.append("-")
        L.append(f"| {t} | " + " | ".join(cells) + " | " + " | ".join(diffs) + " |")
    L += ["\n## Per criterion (means over items), all code-switching tasks and all idiom tasks\n"]
    for kind, tk in (("hm", ("eng_hm", "m_hm")), ("ood", ("eng_ood", "m_ood", "hm_ood"))):
        L += [f"| {kind} criterion | " + " | ".join(LAB[s] for s in SYS) + " | null share |", "|---|" + "---|" * 5]
        for c in crit[kind]:
            vals = {s: [v[s]["sc"].get(c) for (tt, _), v in per.items() if tt in tk and s in v] for s in SYS}
            L.append(f"| {c} | " + " | ".join(f"{np.mean([x for x in vals[s] if ok(x)]):.3f}" if any(ok(x) for x in vals[s]) else "-" for s in SYS) + f" | {1 - np.mean([ok(x) for s in SYS for x in vals[s]]):.2f} |")
        L.append("")
    # win rate of RL latest vs SFT on the overall criterion
    L += ["## Head to head on the overall criterion (ties excluded)\n", "| task | RL latest better | tie | SFT better |", "|---|---|---|---|"]
    for t in TASKS:
        a = [(v["rl_last"]["overall"], v["sft"]["overall"]) for (tt, _), v in per.items() if tt == t and "rl_last" in v and "sft" in v and v["rl_last"]["overall"] is not None and v["sft"]["overall"] is not None]
        if a: L.append(f"| {t} | {sum(x > y for x, y in a) / len(a):.2f} | {sum(x == y for x, y in a) / len(a):.2f} | {sum(x < y for x, y in a) / len(a):.2f} |")
    langs = collections.defaultdict(lambda: collections.defaultdict(list))
    for (tt, item), v in per.items():
        lg = item.split("-")[0]
        for sname in SYS:
            if sname in v and v[sname]["mean"] is not None: langs[lg][sname].append((item, v[sname]["mean"]))
    L += ["\n## Per language: mean criterion score (all tasks) and RL latest - SFT (paired items)\n", "| language | items | SFT | RL best | RL latest | generator | RL latest - SFT |", "|---|---|---|---|---|---|---|"]
    for lg in sorted(langs, key=lambda g: -len(langs[g]["sft"])):
        d = langs[lg]; sm = {sname: dict(d[sname]) for sname in SYS if d[sname]}; both = [i for i in sm.get("rl_last", {}) if i in sm.get("sft", {})]
        L.append(f"| {lg} | {len(sm.get('sft', {}))} | " + " | ".join(f"{np.mean(list(sm[sname].values())):.2f}" if sname in sm else "-" for sname in SYS) + " | " + (f"{np.mean([sm['rl_last'][i] - sm['sft'][i] for i in both]):+.2f}" if both else "-") + " |")
    # prompt sensitivity: old strict compact prompt on the subset
    old = {}
    for kind in ("hm", "ood") if args.old else ():
        for key, rep, um in texts(Path(args.old) / kind / "results" / "stage_s*.raw.jsonl"):
            if rep is None: continue
            v = C.to_vec(kind, rep)
            if v: old[key] = v
    if old:
        L += [f"\n## Prompt sensitivity: old strict compact prompt vs standard v2 on the same items ({len(old)} graded)\n", "| task | system | v2 criterion mean | old prompt dimension mean (H1-H9 or O1-O7) |", "|---|---|---|---|"]; xs = []; ys = []
        for t in TASKS:
            for s in SYS:
                a = []; b = []
                for (tt, item), v in per.items():
                    if tt != t or s not in v or v[s]["mean"] is None: continue
                    ok_key = f"{item.split(':')[0]}|{t}|{s}"
                    if ok_key in old: dims = [x for x in old[ok_key][6:15] + [] if x is not None] if t.endswith("hm") else [x for x in old[ok_key][15:22] if x is not None]
                    else: continue
                    if dims: a.append(v[s]["mean"]); b.append(float(np.mean(dims)))
                if a: L.append(f"| {t} | {LAB[s]} | {np.mean(a):.3f} | {np.mean(b):.3f} |"); xs += a; ys += b
        if len(xs) > 10: L.append(f"\nSpearman between the two prompts across graded (item, system) pairs: {spearmanr(xs, ys)[0]:.3f} (n={len(xs)}).")
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8"); json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open(OUT / "diffs.json", "w"), indent=1)
    json.dump({k: v for k, v in rows.items()}, open(OUT / "scores.json", "w")); print("\n".join(L))


if __name__ == "__main__":
    main()
