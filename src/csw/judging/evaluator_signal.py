"""How well does the evaluator rank outputs for the *same* prompt?

GRPO only uses differences between outputs of one prompt, so a good correlation over the whole test set says little about the training signal.
This compares the evaluator score with the Flash "overall" grade (1) pooled, (2) inside each task, (3) inside each prompt across the three
systems, (4) as pairwise agreement, and (5) fits evaluator -> Flash on the SFT outputs and asks what that predicts for the RL outputs.
The sampled candidates were never judged by Flash, so for those chrF++ stands in.
python -m csw.judging.evaluator_signal --items evaluator_rules_per_item.jsonl.gz --flash scores.json.gz --samples samples_per_item.jsonl.gz --out within_prompt.md"""
import argparse, collections, gzip, json, math

SYS = ["sft", "rl_best", "rl_last"]


def opener(p): return gzip.open(p, "rt", encoding="utf-8") if str(p).endswith(".gz") else open(p, encoding="utf-8")
def mean(x): return sum(x) / len(x)


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i]); out = [0.0] * len(x); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]: j += 1
        for k in range(i, j + 1): out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(a, b): return pearson(ranks(a), ranks(b))


def centred(groups, ia, ib):
    """Subtract each prompt's mean from both scores, so only differences between a prompt's outputs remain."""
    a, b = [], []
    for g in groups:
        if len(g) < 3: continue
        ma, mb = mean([r[ia] for r in g]), mean([r[ib] for r in g])
        a += [r[ia] - ma for r in g]; b += [r[ib] - mb for r in g]
    return a, b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--items", required=True); ap.add_argument("--flash", required=True); ap.add_argument("--samples", default=""); ap.add_argument("--out", required=True); args = ap.parse_args()
    flash = json.load(opener(args.flash)); rows = []                                           # (id, system, task, evaluator, reward, flash overall)
    for l in opener(args.items):
        r = json.loads(l); f = flash.get(f"{r['id']}|{r['system']}")
        if r["system"] in SYS and f and f.get("overall") is not None and r["evaluator"] is not None and r["reward"] is not None:
            rows.append((r["id"], r["system"], r["task"], r["evaluator"], r["reward"], f["overall"]))
    by = collections.defaultdict(list)
    for r in rows: by[r[0]].append(r)
    L = [f"# Evaluator signal against Flash (overall criterion), {len(rows)} outputs, {len(by)} prompts\n"]
    ev, rw, fl = [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows]
    a, b = centred(by.values(), 3, 5); c, d = centred(by.values(), 4, 5)
    L += ["| view | evaluator | reward |", "|---|---|---|",
          f"| pooled over all outputs, Spearman | {spearman(ev, fl):+.3f} | {spearman(rw, fl):+.3f} |",
          f"| inside one prompt (3 systems), Pearson | {pearson(a, b):+.3f} | {pearson(c, d):+.3f} |"]
    win = lose = tie = 0
    for g in by.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                x, y = g[i], g[j]
                if x[5] == y[5]: tie += 1; continue
                if x[3] == y[3]: continue
                if (x[3] > y[3]) == (x[5] > y[5]): win += 1
                else: lose += 1
    L += [f"\nPairwise: when Flash prefers one output the evaluator prefers the same one {100 * win / (win + lose):.1f}% of the time ({win + lose} pairs; {tie} pairs tied for Flash were skipped).\n",
          "| task | evaluator mean | Flash overall mean | pooled Spearman in task | inside-prompt Pearson |", "|---|---|---|---|---|"]
    for t in sorted({r[2] for r in rows}):
        sub = [r for r in rows if r[2] == t]; g = collections.defaultdict(list)
        for r in sub: g[r[0]].append(r)
        x, y = centred(g.values(), 3, 5)
        L.append(f"| {t} | {mean([r[3] for r in sub]):.2f} | {mean([r[5] for r in sub]):.2f} | {spearman([r[3] for r in sub], [r[5] for r in sub]):+.3f} | {pearson(x, y):+.3f} |")
    sft = sorted([r for r in rows if r[1] == "sft"], key=lambda r: r[3]); nb = 20
    edges = [sft[int(len(sft) * k / nb)][3] for k in range(nb)] + [1e9]
    table = [mean([r[5] for r in sft if edges[k] <= r[3] < edges[k + 1]]) for k in range(nb)]
    def predict(e):
        for k in range(nb):
            if edges[k] <= e < edges[k + 1]: return table[k]
        return table[0]
    mx, my = mean([r[3] for r in sft]), mean([r[5] for r in sft])
    slope = sum((r[3] - mx) * (r[5] - my) for r in sft) / sum((r[3] - mx) ** 2 for r in sft)
    L += ["\nCalibration fitted on the SFT outputs (20 evaluator quantile bins -> mean Flash overall), applied to every system:\n", "| system | evaluator | Flash predicted from evaluator | Flash actual | actual - predicted |", "|---|---|---|---|---|"]
    for s in SYS:
        sub = [r for r in rows if r[1] == s]; p, act = mean([predict(r[3]) for r in sub]), mean([r[5] for r in sub])
        L.append(f"| {s} | {mean([r[3] for r in sub]):.3f} | {p:.3f} | {act:.3f} | {act - p:+.3f} |")
    L.append(f"\nOn the SFT outputs one evaluator point is worth {slope:.2f} Flash points; mean Flash minus evaluator {my - mx:+.2f}. A constant offset cancels in GRPO (advantages are differences inside a group), so this bias is irrelevant to the update.")
    if args.samples:
        a, b, n = [], [], 0
        for l in opener(args.samples):
            r = json.loads(l)
            if r["system"] != "rl_best" or len(r["ev"]) < 3: continue
            me, mc = mean(r["ev"]), mean(r["chrf"]); a += [x - me for x in r["ev"]]; b += [y - mc for y in r["chrf"]]; n += 1
        L.append(f"\nSampled groups (RL best, {n} prompts, 9 candidates each, no Flash grades): inside-prompt Pearson between evaluator and chrF++ {pearson(a, b):+.3f}.")
    open(args.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
