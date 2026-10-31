"""Fresh RL prompt pool: NO source text from the SFT data (train / val / test).
  eng_hm / eng_ood / eng_m : fresh English (english_fresh.jsonl, minus english_test.jsonl), every sentence used once.
  m_hm / m_ood            : base-language sources = full_E's own greedy eng_m translations of OTHER fresh English sentences (so they cannot occur in the SFT data), filtered (LID, length, loop, English leak, format).
  hm_ood                  : sources = full_E's greedy eng_hm text (devices removed) of other fresh English sentences, filtered the same way.
  python -m csw.rl.build_fresh plan --old rl_prompts --english english_fresh.jsonl --test-english english_test.jsonl --out fresh_work --total 52000
  (generate fresh_work/gen_in.jsonl with csw.sft.gen_vllm --samples 1 -> fresh_work/gen_out.jsonl)
  python -m csw.rl.build_fresh assemble --work fresh_work --sft final_v5 --out rl_prompts_fresh"""
import argparse, collections, hashlib, json, random, re, sys
from pathlib import Path
from csw.sft import fmt2
from csw.rl import rules as R

TASK_MIX = {"eng_hm": 0.20, "m_hm": 0.15, "eng_ood": 0.20, "m_ood": 0.15, "hm_ood": 0.15, "eng_m": 0.15}
sha = lambda s: hashlib.sha1(" ".join((s or "").lower().split()).encode()).hexdigest()
norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
key = lambda t: hashlib.md5(norm(t)[:400].encode()).hexdigest()


def plan(a):
    rnd = random.Random(a.seed); held = {sha(json.loads(l).get("english")) for l in open(a.test_english, encoding="utf-8")}
    fresh = [json.loads(l) for l in open(a.english, encoding="utf-8")]; fresh = [r for r in fresh if sha(r["english"]) not in held and 4 <= len(r["english"].split()) <= a.max_words]; rnd.shuffle(fresh)
    combos = collections.defaultdict(list)
    for name in ("train.jsonl", "monitor.jsonl"):
        for l in open(Path(a.old) / name, encoding="utf-8"):
            r = json.loads(l); sw = r.get("switch"); combos[r["task"]].append((r["lang"], None if sw in (None, "None") else sw))
    fi = 0; gen_in, eng_items = [], []; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for task, w in TASK_MIX.items():
        n = round(a.total * w)
        for k in range(n):
            lang, sw = rnd.choice(combos[task]); e = fresh[fi]; fi += 1
            if task in ("eng_hm", "eng_ood", "eng_m"):
                eng_items.append({"id": f"rf-{task}-{k}", "task": task, "lang": lang, "switch": sw if task == "eng_hm" else None, "prompt": fmt2.prompt(task, lang, sw if task == "eng_hm" else None, e["english"]), "source": e["english"], "eng": e["english"]})
            else:
                gtask = "eng_hm" if task == "hm_ood" else "eng_m"; gsw = sw if task == "hm_ood" else None
                gen_in.append({"id": f"rf-{task}-{k}", "lang": lang, "task": gtask, "target_task": task, "switch": sw, "prompt": fmt2.prompt(gtask, lang, gsw, e["english"]), "completion": "", "english": e["english"]})
    with open(out / "gen_in.jsonl", "w", encoding="utf-8") as f:
        for r in gen_in: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "eng_items.jsonl", "w", encoding="utf-8") as f:
        for r in eng_items: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("english used", fi, "of", len(fresh), "| generation items", len(gen_in), "| english-based prompts", len(eng_items))


def assemble(a):
    from csw.rl.lid import Lid, sentences, allowed
    work = Path(a.work); lid = Lid(); rnd = random.Random(a.seed); sft = set()
    for sp in ("train", "val", "test_off", "test_on"):
        for l in open(Path(a.sft) / f"{sp}.jsonl", encoding="utf-8"):
            r = json.loads(l); sft.add(key(r["prompt"].split(":", 1)[1] if ":" in r["prompt"] else r["prompt"]))
    gens = {}
    for l in open(work / "gen_out.jsonl", encoding="utf-8"):
        r = json.loads(l); gens[r["id"]] = r
    keep, drop = [], collections.Counter()
    for l in open(work / "gen_in.jsonl", encoding="utf-8"):
        e = json.loads(l); g = gens.get(e["id"])
        if not g: drop["missing"] += 1; continue
        p = R.parse(g["generations"][0] + " <end_of_turn>"); fin = p["final"]
        if e["task"] == "eng_m": fin = re.sub(r"^[a-z]{3}\s+", "", fin)
        tt = e["target_task"]
        if not fin or not p["eos_ok"]: drop["format"] += 1; continue
        if e["task"] == "eng_hm" and not p["dev_ok"]: drop["format"] += 1; continue
        r = len(fin) / max(1, len(e["english"]))
        if not (0.5 <= r <= 2.2): drop["length"] += 1; continue
        if R.loop_penalty(fin)[0] > 0: drop["loop"] += 1; continue
        if tt != "hm_ood" and R.english_leak(fin, e["lang"] in ("pcm", "gpe", "wes", "afr"), cap=0.10)[0] > 0: drop["english"] += 1; continue
        if tt == "hm_ood": fin = R.strip_devices(fin, []) or fin
        if R.number_penalty(fin, e["english"])[0] >= 0.5: drop["number"] += 1; continue
        ss = sentences(fin)
        if ss:
            labs = lid.label(ss); share = sum(x in allowed(e["lang"]) for x in labs) / len(labs)
            if share < (0.6 if tt != "hm_ood" else 0.3): drop["lid"] += 1; continue
        if key(fin) in sft: drop["in_sft"] += 1; continue
        keep.append({"id": e["id"], "task": tt, "lang": e["lang"], "switch": e["switch"] if tt != "m_ood" else None, "prompt": fmt2.prompt(tt, e["lang"], e["switch"] if tt != "m_ood" else None, fin), "source": fin, "eng": None})
    items = [json.loads(l) for l in open(work / "eng_items.jsonl", encoding="utf-8")] + keep; rnd.shuffle(items)
    ev, tr, cnt = [], [], collections.Counter()
    for it in items:
        if cnt[it["task"]] < a.eval_per_task: ev.append(it); cnt[it["task"]] += 1
        else: tr.append(it)
    ov = sum(key(it["source"]) in sft for it in items); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", tr), ("monitor.jsonl", ev)):
        with open(out / name, "w", encoding="utf-8") as f:
            for it in rows: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("dropped", dict(drop), "| train", len(tr), "monitor", len(ev), dict(collections.Counter(i["task"] for i in tr)), "| overlap with SFT sources:", ov)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--old", required=True); p.add_argument("--english", required=True); p.add_argument("--test-english", required=True); p.add_argument("--out", required=True); p.add_argument("--total", type=int, default=52000); p.add_argument("--max-words", type=int, default=90); p.add_argument("--seed", type=int, default=11)
    q = sub.add_parser("assemble"); q.add_argument("--work", required=True); q.add_argument("--sft", required=True); q.add_argument("--out", required=True); q.add_argument("--eval-per-task", type=int, default=120); q.add_argument("--seed", type=int, default=12)
    a = ap.parse_args(); plan(a) if a.cmd == "plan" else assemble(a)


if __name__ == "__main__":
    main()
