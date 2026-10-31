"""RL prompt pool for the csw model.  eng_* tasks use FRESH English (never in the SFT data: english_fresh.jsonl, minus every sentence of english_test.jsonl); m_hm / m_ood / hm_ood use the
sources of SFT training rows (the model's own translations / code-switched texts exist only there).  Languages are balanced with a cap per language (UniMax-like draw weights), tasks by TASK_MIX.
python -m csw.rl.build_prompts --sft final_v5/C/train.jsonl --english english_fresh.jsonl --test-english english_test.jsonl --out rl_prompts --per-lang 600"""
import argparse, collections, hashlib, json, random, sys
from pathlib import Path
from csw.sft import fmt2

TASK_MIX = {"eng_hm": 0.20, "m_hm": 0.15, "eng_ood": 0.20, "m_ood": 0.15, "hm_ood": 0.15, "eng_m": 0.15}
sha = lambda s: hashlib.sha1(" ".join((s or "").lower().split()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sft", required=True); ap.add_argument("--english", required=True); ap.add_argument("--test-english", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=600); ap.add_argument("--max-words", type=int, default=90); ap.add_argument("--eval-per-task", type=int, default=120); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    rnd = random.Random(a.seed); held = {sha(json.loads(l).get("english")) for l in open(a.test_english, encoding="utf-8")}
    fresh = [json.loads(l) for l in open(a.english, encoding="utf-8")]; fresh = [r for r in fresh if sha(r["english"]) not in held and 4 <= len(r["english"].split()) <= a.max_words]; rnd.shuffle(fresh); print("fresh English", len(fresh))
    combos = collections.defaultdict(lambda: collections.defaultdict(collections.Counter)); pool = collections.defaultdict(list)
    for l in open(a.sft, encoding="utf-8"):
        r = json.loads(l); t = r.get("task"); lang = r["lang"]; sw = r.get("switch") or None
        if t in ("eng_hm", "eng_ood", "eng_m", "m_hm"): combos[t][lang][sw] += 1
        if t in ("m_hm", "m_ood", "hm_ood") and 4 <= len(r["prompt"].split(":", 1)[-1].split()) <= a.max_words: pool[(t, lang)].append(r)
    langs = sorted({l for t in combos for l in combos[t]} | {l for (_, l) in pool}); items = []; fi = 0
    for lang in langs:
        for task, w in TASK_MIX.items():
            n = max(1, round(a.per_lang * w))
            if task in ("eng_hm", "eng_ood", "eng_m"):
                sws = combos[task if task != "eng_ood" else "eng_ood"][lang] or combos["eng_hm"][lang]
                if not sws: continue
                keys = list(sws); wts = [sws[k] for k in keys]
                for _ in range(n):
                    e = fresh[fi % len(fresh)]; fi += 1; sw = rnd.choices(keys, wts)[0] if task == "eng_hm" else (None if task != "eng_hm" else sw)
                    items.append({"id": f"rl-{task}-{lang}-{fi}", "task": task, "lang": lang, "switch": sw if task == "eng_hm" else None, "prompt": fmt2.prompt(task, lang, sw if task == "eng_hm" else None, e["english"]), "source": e["english"], "eng": e["english"]})
            else:
                rows = pool[(task, lang)]; rnd.shuffle(rows)
                for r in rows[:n]:
                    items.append({"id": "rl-" + r["id"], "task": task, "lang": lang, "switch": r.get("switch") if task != "m_ood" else None, "prompt": r["prompt"], "source": r["prompt"].split(":", 1)[1].strip(), "eng": None})
    rnd.shuffle(items); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev, tr, cnt = [], [], collections.Counter()
    for it in items:                                                         # a fixed monitor set per task, never trained on
        if cnt[it["task"]] < a.eval_per_task: ev.append(it); cnt[it["task"]] += 1
        else: tr.append(it)
    for name, rows in (("train.jsonl", tr), ("monitor.jsonl", ev)):
        with open(out / name, "w", encoding="utf-8") as f:
            for it in rows: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("train", len(tr), "monitor", len(ev), dict(collections.Counter(i["task"] for i in tr)), "languages", len(langs))


if __name__ == "__main__":
    main()
