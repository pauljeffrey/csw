"""Standard system-blind judge (v2) for the SFT vs RL comparison on the SFT test set; Gemini 3.8 Flash, Batch API only.  python -m csw.judging.judge_v2 [--per-task 200] [--shard 750]"""
import argparse, collections, glob, json, random, sys
from pathlib import Path
from csw.rl import rules as R
from csw.judging import judge_v11_test as J

from csw.paths import DATA_ROOT
W = DATA_ROOT / "work/judge_v2"; A = DATA_ROOT / "work/system_comparison"; TASKS = ("eng_hm", "m_hm", "eng_ood", "m_ood", "hm_ood")
SYSTEMS = {"sft": A / "gens_sft.jsonl", "rl_best": A / "gens_rl_best.jsonl", "rl_last": A / "gens_rl_last.jsonl", "reference": None}   # reference = the generator model's curated text of the test set
OPEN = "You are an expert evaluator of written African-language text, with native-level knowledge of {langs}.\n"
ANCH = ("Anchors (every criterion): 5 = fully correct, a native speaker would change nothing; 4 = minor issues that do not affect understanding; 3 = noticeable issues; 2 = serious issues that affect meaning or naturalness; 1 = unusable or wrong.\n"
        "Rate the OUTPUT only; ignore its length. Rate each criterion independently, using only what is written in the INPUT and OUTPUT. If you cannot judge a criterion in this language, give null; do not guess.\n")
SYS_HM = (OPEN + "A text-generation system was given an INPUT and asked to rewrite it as natural code-switched text that mixes {language} (the base language) with {switch_language}. You will see the INPUT, the meaning of the input in English, and the system's OUTPUT.\n" + ANCH +
          "Criteria:\n- meaning: the OUTPUT conveys all the information of the INPUT, adds nothing and changes nothing.\n- base_language: grammar, word choice, spelling and orthography (including diacritics and special letters) of the {language} parts.\n"
          "- switch_language: grammar, word choice and spelling of the {switch_language} parts.\n- language_compliance: every word is in {language} or {switch_language}; 5 = no other language appears, 1 = a third language or untranslated material dominates.\n"
          "- switching_naturalness: switches happen at points where real bilingual speakers switch, in units that fit the grammar; 1 = random, forced or word-by-word substitution.\n- overall: your overall judgement of the OUTPUT as a piece of code-switched text.\n"
          'Reply with one JSON object and nothing else: {{"scores": {{"meaning": n, "base_language": n, "switch_language": n, "language_compliance": n, "switching_naturalness": n, "overall": n}}, "issues": ["at most 3 short strings naming the main problems, with the exact words concerned"]}}')
SYS_OOD = (OPEN + "A text-generation system was given an INPUT and asked to rewrite it as informal, idiomatic {language}, using idioms, proverbs or expressions that native speakers really use. You will see the INPUT, the meaning of the input in English, and the system's OUTPUT.\n" + ANCH +
           "Criteria:\n- meaning: the OUTPUT conveys all the information of the INPUT, adds nothing and changes nothing.\n- language_correctness: grammar, word choice, spelling and orthography (including diacritics) of the OUTPUT.\n"
           "- idiom_use: the idioms or expressions used are real, apt and correctly used; null if you cannot verify them; do not call an expression wrong only because you do not know it.\n"
           "- naturalness: the register and phrasing sound like the informal speech of a native speaker; 1 = stiff, literal or odd.\n- language_compliance: the OUTPUT is in {language}; occasional words of {allowed} are acceptable; 1 = another language dominates.\n"
           "- overall: your overall judgement of the OUTPUT as informal idiomatic text.\n"
           'Reply with one JSON object and nothing else: {{"scores": {{"meaning": n, "language_correctness": n, "idiom_use": n, "naturalness": n, "language_compliance": n, "overall": n}}, "issues": ["at most 3 short strings naming the main problems, with the exact words concerned"]}}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-task", type=int, default=200); ap.add_argument("--shard", type=int, default=750); ap.add_argument("--show", action="store_true"); ap.add_argument("--full", action="store_true", help="all test items instead of a stratified sample"); ap.add_argument("--systems", default="sft,rl_best,rl_last,reference"); a = ap.parse_args(); want = a.systems.split(",")
    raw = {}
    for f in glob.glob(str(DATA_ROOT / "work/post_rows/*.raw.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); raw[r["id"]] = r
    tests = {}
    for l in open(A / "test_off.jsonl", encoding="utf-8"):
        t = json.loads(l); tests[t["id"]] = t
    gens = {s: {} for s in SYSTEMS}
    gens["reference"] = {i: {"generations": [t["completion"]]} for i, t in tests.items()}
    for s, f in SYSTEMS.items():
        if f is not None and f.exists():
            for l in open(f, encoding="utf-8"):
                g = json.loads(l); gens[s][g["id"]] = g
    have = [s for s in want if gens[s]]; by = collections.defaultdict(list)
    for tid, t in tests.items():
        r = raw.get(tid.split(":")[0])
        if t["task"] not in TASKS or not r or not isinstance(r.get("eng"), str) or not r["eng"]: continue
        if all((gens[s].get(tid) and R.parse(gens[s][tid]["generations"][0] + " <end_of_turn>")["final"]) for s in have): by[t["task"]].append(t)
    rnd = random.Random(31); chosen = []
    for task in TASKS:
        bl = collections.defaultdict(list)
        for t in by[task]: bl[t["lang"]].append(t)
        langs = sorted(bl); rnd.shuffle(langs); [rnd.shuffle(v) for v in bl.values()]; pick = []; i = 0
        while len(pick) < a.per_task and any(bl.values()):
            l = langs[i % len(langs)]; i += 1
            if bl[l]: pick.append(bl[l].pop())
        chosen += (pick if not a.full else by[task])
    reqs = []
    for t in chosen:
        raw_row = raw[t["id"].split(":")[0]]; lang = t["lang"]; P = J.P[lang]; sw = t.get("switch"); hm = t["task"].endswith("hm")
        swn = J.P[sw].name if sw in J.P else {"eng": "English", "fra": "French", "ara": "Arabic", "pcm": "Nigerian Pidgin", "sheng": "Sheng"}.get(sw, sw)
        for s in have:
            p = R.parse(gens[s][t["id"]]["generations"][0] + " <end_of_turn>")
            sysm = (SYS_HM.format(langs=f"{P.name} and {swn}", language=P.name, switch_language=swn) if hm else SYS_OOD.format(langs=P.name, language=P.name, allowed=", ".join(P.switch) or "none"))
            msg = {"language": P.name, **({"switch_language": swn} if hm else {}), "input": t["prompt"].split(":", 1)[1].strip(), "meaning_in_english": raw_row["eng"], "output": p["final"]}
            reqs.append({"key": f"{t['id']}|{s}", "task": t["task"], "system": s, "sys": sysm, "user": json.dumps(msg, ensure_ascii=False)})
    rnd.shuffle(reqs)                                                                      # systems and tasks are mixed: no judge sees a block of one system
    if a.show:
        for task in ("eng_hm", "eng_ood"):
            q = next(r for r in reqs if r["task"] == task); print(f"===== SYSTEM PROMPT ({task}) =====\n{q['sys']}\n\n===== USER MESSAGE =====\n{q['user']}\n")
    d = W / ("requests_full" if a.full else "requests"); d.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(reqs), a.shard):
        sd = d / f"stage_s{k // a.shard:02d}"; sd.mkdir(exist_ok=True)
        (sd / "gemini.jsonl").write_text("\n".join(json.dumps({"key": r["key"], "request": {"contents": [{"role": "user", "parts": [{"text": r["user"]}]}], "system_instruction": {"parts": [{"text": r["sys"]}]},
                                      "generation_config": {"temperature": 0, "max_output_tokens": 800, "response_mime_type": "application/json", "thinking_config": {"thinking_level": "low"}}}}, ensure_ascii=False) for r in reqs[k:k + a.shard]) + "\n", encoding="utf-8")
    json.dump([{k: r[k] for k in ("key", "task", "system")} for r in reqs], open(W / ("meta_full.json" if a.full else "meta.json"), "w")); json.dump([t["id"] for t in chosen], open(W / ("chosen_full.json" if a.full else "chosen.json"), "w"))
    chars = sum(len(r["sys"]) + len(r["user"]) for r in reqs); print(f"systems {have}; items {len(chosen)}; requests {len(reqs)}; shards {1 + (len(reqs) - 1) // a.shard}; approx input tokens {int(chars / 3.2):,}")
    tin = chars / 3.2
    for out in (120, 250, 450): print(f"  Flash batch ($0.375 in / $1.875 out per 1M): {out} output tokens/request -> ${tin * 0.375e-6 + len(reqs) * out * 1.875e-6:.2f}")


if __name__ == "__main__":
    main()
