"""Argon head-to-head: SFT model vs the best and the last RL checkpoint on the same held-out SFT-test items (greedy outputs).
For every selected test item the system's output replaces the hm (or ood) field of the item's raw generator row, and the dedicated strict compact judge (cs_sft.compact_judge) grades it:
hm tasks -> H1..H9, D1, PD; ood tasks -> O1..O7, D2, IQ.   python -m csw.judging.judge_prompt_sensitivity [--per-task 60]"""
import argparse, collections, copy, glob, json, random, sys
from pathlib import Path
from csw.rl import rules as R
from csw.judging import compact_judge as C, devices as dv

from csw.paths import DATA_ROOT
W = DATA_ROOT / "work/system_comparison"; TASKS = ("eng_hm", "m_hm", "eng_ood", "m_ood", "hm_ood"); SYSTEMS = {"sft": "gens_sft.jsonl", "rl_best": "gens_rl_best.jsonl", "rl_last": "gens_rl_last.jsonl", "reference": None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-task", type=int, default=60); ap.add_argument("--max-out", type=int, default=1500); ap.add_argument("--systems", default="sft,rl_best,rl_last"); a = ap.parse_args(); want = a.systems.split(",")
    raw = {}
    for f in glob.glob(str(DATA_ROOT / "work/post_rows/*.raw.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); raw[r["id"]] = r
    gens = {s: {} for s in SYSTEMS}
    for s, f in SYSTEMS.items():
        if f is None: continue
        if not (W / f).exists(): print("missing", f); continue
        for l in open(W / f, encoding="utf-8"):
            g = json.loads(l); gens[s][g["id"]] = g
    gens["reference"] = {t["id"]: {"generations": [t["completion"]]} for t in map(json.loads, open(W / "test_off.jsonl", encoding="utf-8"))}
    have = [s for s in want if gens[s]]; print("systems with outputs:", have)
    tests = [json.loads(l) for l in open(W / "test_off.jsonl", encoding="utf-8")]; by = collections.defaultdict(list); fixed = json.load(open(W / "chosen.json")) if (W / "chosen.json").exists() else None
    for t in tests:
        if t["task"] not in TASKS: continue
        root = t["id"].split(":")[0]; r = raw.get(root)
        if not r or not all(isinstance(r.get(k), str) and r.get(k) for k in ("eng", "hm", "ood")): continue
        ok = True
        if fixed is not None: ok = t["id"] in fixed
        for s in ([] if fixed is not None else have):
            g = gens[s].get(t["id"]); p = R.parse(g["generations"][0] + " <end_of_turn>") if g else None
            if not p or not p["final"]: ok = False
        if ok: by[t["task"]].append(t)
    rnd = random.Random(2026); chosen = []
    if fixed is not None:
        chosen = [t for task in TASKS for t in by[task]]; print("reusing the fixed item set:", len(chosen))
    for task in (TASKS if fixed is None else ()):
        bl = collections.defaultdict(list)
        for t in by[task]: bl[t["lang"]].append(t)
        langs = sorted(bl); rnd.shuffle(langs); pick = []; i = 0
        for l in langs: rnd.shuffle(bl[l])
        while len(pick) < a.per_task and any(bl.values()):
            l = langs[i % len(langs)]; i += 1
            if bl[l]: pick.append(bl[l].pop())
        chosen += pick; print(task, "eligible", len(by[task]), "chosen", len(pick))
    if fixed is None: json.dump([t["id"] for t in chosen], open(W / "chosen.json", "w"))
    rows = {"hm": [], "ood": []}; meta = []
    for t in chosen:
        root = t["id"].split(":")[0]
        for s in have:
            g = gens[s].get(t["id"]); p = R.parse(g["generations"][0] + " <end_of_turn>") if g else None
            if not p or not p["final"]: meta.append({"id": f"{root}|{t['task']}|{s}", "item": t["id"], "task": t["task"], "lang": t["lang"], "system": s, "no_output": True}); continue
            r = copy.deepcopy(raw[root]); r["id"] = f"{root}|{t['task']}|{s}"; kind = "hm" if t["task"].endswith("hm") else "ood"
            if kind == "hm": r["hm"] = p["final"]
            else: r["ood"] = p["final"]; r["devices"] = p["devices"]
            rows[kind].append(r); meta.append({"id": r["id"], "item": t["id"], "task": t["task"], "lang": t["lang"], "system": s, "kind": kind, "format_ok": bool(p["dev_ok"])})
    lex = dv.Lexicon.load(str(DATA_ROOT / "work/lexicon.json"))
    tag = "_".join(have)
    for kind in ("hm", "ood"):
        d = W / tag / kind; (d / "stage_s00").mkdir(parents=True, exist_ok=True); sysm = {"hm": C.SYS_HM, "ood": C.SYS_OOD}[kind]; usr = {"hm": C.user_hm, "ood": C.user_ood}[kind]; lines = []
        with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
            for r in rows[kind]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in rows[kind]:
            lines.append({"key": r["id"], "request": {"contents": [{"role": "user", "parts": [{"text": usr(r, lex)}]}], "system_instruction": {"parts": [{"text": sysm}]},
                                                      "generation_config": {"temperature": 0, "max_output_tokens": a.max_out, "response_mime_type": "application/json", "thinking_config": {"thinking_level": "low"}}}})
        (d / "stage_s00" / "gemini.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8"); print(kind, "requests", len(lines))
    json.dump(meta, open(W / f"meta_{tag}.json", "w"), ensure_ascii=False)
    chars = sum(len(json.dumps(x)) for k in ("hm", "ood") for x in [json.loads(l) for l in open(W / tag / k / "stage_s00" / "gemini.jsonl")]); print("approx input tokens (chars/3.2):", int(chars / 3.2))


if __name__ == "__main__":
    main()
