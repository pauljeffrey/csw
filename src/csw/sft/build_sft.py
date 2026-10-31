"""Build the SFT set for the code-switcher (eng -> m_text | hm | ood) from the generated rows (+ gold).

python -m csw.sft.build_sft --rows DIR_WITH_*.raw.jsonl --out OUT_DIR [--gold ../exemplars/augmented_gold.jsonl] [--cap-per-lang-task 0] [--pidgin-cap 0]

Per row it can write up to three examples (one per task).  Filters (all reported in stats.json):
  - gates.check fail                                  -> row dropped entirely (empty field, English leak, wrong script, hm == m_text, ...)
  - hm task needs >=1 recovered switch phrase and a switch ratio in [0.04, 0.65], and no 'english_when_not_pinned' / 'hm_over_mixed' / 'hm_repetitive' warn
  - ood task needs >=2 reported idioms that really occur in the ood text
  - markdown asterisks (*word*) are removed, text is NFC
Split: by sha1(english) so one English sentence never appears in train and eval; 2% val, 2% test (gold rows are train-only).
Output files: sft_train.jsonl, sft_val.jsonl, sft_test.jsonl, stats.json, lexicon.json. Each line has prompt, completion, text (Gemma chat string), lang, task, base/switch tags and meta.
"""
from __future__ import annotations

from csw.paths import DATA_ROOT
import argparse
import collections
import glob
import hashlib
import json
import sys
from pathlib import Path

HERE = DATA_ROOT
from csw.datagen import gates                                  # noqa: E402
from csw.datagen.profiles import P, TARGETS, HOLD         # noqa: E402
from csw.sft import fmt         # noqa: E402
from csw.sft import devices as dv         # noqa: E402

DROP_WARN_HM = {"english_when_not_pinned", "hm_over_mixed", "hm_repetitive", "pidgin_leak_in_hm"}
OOD_SWITCH_MIN = 0.08          # the second language tag is written for ood only when this much of the ood text is switch-language
MAX_CHARS = 3200


def clean(t: str) -> str:
    return " ".join(dv.norm(t).split())


def load_rows(rows_dir: str, gold: list[str]) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(Path(rows_dir) / "*.raw.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if all(r.get(k) for k in ("eng", "m_text", "hm", "ood")) and not r.get("error"):
                    r["source"] = "generated"; out.append(r)
    for g in gold:
        for line in open(g, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if all(r.get(k) for k in ("eng", "m_text", "hm", "ood")) and r.get("lang"):
                    r["scenario"] = {"switch": r.get("switch") or "English", "dialect": r.get("dialect"), "genre": r.get("genre")}
                    r["id"] = "gold-" + hashlib.sha1((r["eng"] + r["m_text"]).encode()).hexdigest()[:10]
                    r["source"] = "gold"; out.append(r)
    return out


def split_of(eng: str, source: str) -> str:
    if source == "gold":
        return "train"
    h = int(hashlib.sha1(eng.strip().lower().encode()).hexdigest(), 16) % 100
    return "val" if h < 2 else ("test" if h < 4 else "train")


def examples(r: dict, lex: dv.Lexicon, st: collections.Counter) -> list[dict]:
    lang = r["lang"]
    if lang not in fmt.NAMES or lang not in P:
        st["skip_lang_not_in_set"] += 1; return []
    for k in ("eng", "m_text", "hm", "ood"):
        r[k] = clean(r[k])
        if len(r[k]) > MAX_CHARS:
            st["skip_too_long"] += 1; return []
    g = gates.check(r, P[lang]) if r.get("scenario", {}).get("length_spec") or r["source"] == "gold" else {"fail": [], "warn": []}
    if r["source"] == "gold":
        g = {"fail": [], "warn": []}               # gold rows were corrected one by one; their older sentence-count bands must not exclude them
    if g["fail"]:
        st["row_dropped_gate_fail"] += 1
        for f in g["fail"]:
            st["fail:" + f] += 1
        return []
    sw_name = r["scenario"]["switch"]
    sw = fmt.SWITCH_TAG.get(sw_name)
    if sw is None:
        st["skip_unknown_switch"] += 1; return []
    meta = {"id": r["id"], "source": r["source"], "genre": (r.get("scenario") or {}).get("genre"), "dialect": (r.get("scenario") or {}).get("dialect"),
            "warn": g.get("warn", [])}
    split = split_of(r["eng"], r["source"])
    out = []

    def add(task, prompt, completion, extra):
        out.append({"id": f"{r['id']}:{task}", "lang": lang, "task": task, "split": split, "source": r["source"], "prompt": prompt, "completion": completion,
                    "text": fmt.gemma_text(prompt, completion), "eng": r["eng"], "meta": {**meta, **extra}})

    add("m_text", fmt.prompt_m_text(lang, r["eng"]), fmt.completion_m_text(lang, r["m_text"]), {})
    st["ex:m_text"] += 1
    h = dv.hm_devices(r, lex)
    if not h["devices"]:
        st["hm_dropped_no_switch_found"] += 1
    elif not 0.04 <= h["switch_ratio"] <= 0.65:
        st["hm_dropped_ratio"] += 1
    elif set(g.get("warn", [])) & DROP_WARN_HM:
        st["hm_dropped_warn"] += 1
    else:
        add("hm", fmt.prompt_hm(lang, sw, r["eng"]), fmt.completion_hm(lang, sw, h["devices"], r["hm"]), {"devices": h["devices"], "switch_ratio": h["switch_ratio"], "switch": sw})
        st["ex:hm"] += 1
    o = dv.ood_devices(r)
    if len(o["devices"]) < 2:
        st["ood_dropped_few_devices"] += 1
    else:
        osw = dv.hm_devices(r, lex, field="ood")["switch_ratio"]
        tag2 = sw if osw >= OOD_SWITCH_MIN else None
        add("ood", fmt.prompt_ood(lang, r["eng"]), fmt.completion_ood(lang, o["devices"], r["ood"], tag2),
            {"devices": o["devices"], "glossed": o["glossed"], "switch_ratio": osw, "switch": tag2})
        st["ex:ood"] += 1
        st["ood_second_tag"] += tag2 is not None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gold", nargs="*", default=[str(HERE / "exemplars" / "augmented_gold.jsonl")])
    ap.add_argument("--cap-per-lang-task", type=int, default=0, help="max train examples per (language, task); 0 = no cap")
    ap.add_argument("--pidgin-cap", type=int, default=0, help="cap per task for pcm/gpe/wes (they finished first and are over-represented); 0 = no cap")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(a.rows, [g for g in a.gold if Path(g).exists()])
    seen, uniq = set(), []
    for r in rows:
        k = (r["lang"], r["eng"].strip().lower(), r["m_text"].strip().lower())
        if k not in seen:
            seen.add(k); uniq.append(r)
    st = collections.Counter(); st["rows_in"] = len(rows); st["rows_after_dedupe"] = len(uniq)
    lex = dv.Lexicon.build([r for r in uniq if r["source"] == "generated"]); lex.save(out / "lexicon.json")
    all_ex = []
    for r in uniq:
        all_ex += examples(r, lex, st)
    import random
    rnd = random.Random(0)
    by = collections.defaultdict(list)
    for e in all_ex:
        by[(e["lang"], e["task"], e["split"])].append(e)
    final = []
    for (lang, task, split), v in by.items():
        if split == "train":
            cap = a.pidgin_cap if lang in ("pcm", "gpe", "wes") else a.cap_per_lang_task
            if cap and len(v) > cap:
                gold = [x for x in v if x["source"] == "gold"]; rest = [x for x in v if x["source"] != "gold"]
                rnd.shuffle(rest); v = gold + rest[: max(0, cap - len(gold))]
        final += v
    rnd.shuffle(final)
    for split in ("train", "val", "test"):
        with open(out / f"sft_{split}.jsonl", "w", encoding="utf-8") as f:
            for e in final:
                if e["split"] == split:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
    cnt = collections.Counter((e["split"], e["task"]) for e in final)
    per = collections.defaultdict(collections.Counter)
    for e in final:
        if e["split"] == "train":
            per[e["lang"]][e["task"]] += 1
    stats = {"counters": dict(st), "examples_by_split_task": {f"{s}/{t}": n for (s, t), n in sorted(cnt.items())}, "train_by_lang_task": {l: dict(c) for l, c in sorted(per.items())}}
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({"rows": st["rows_after_dedupe"], "examples": stats["examples_by_split_task"]}, indent=1))


if __name__ == "__main__":
    main()
