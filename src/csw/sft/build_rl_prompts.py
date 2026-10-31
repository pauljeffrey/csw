"""RL (GRPO) prompt sets for the code-switcher: English source + task + base/switch language; NO targets (the reward is the QE + csw.sft.rewards).

python -m csw.sft.build_rl_prompts --english english_fresh.jsonl --out OUT_DIR [--per-lang 10000] [--per-lang-strong 20000] [--seed 0]
(--rows DIR  instead of --english reuses the English of the generated rows = the v0 behaviour; RL should use FRESH English)

One English statement can be asked for many languages and tasks, so the English pool (every distinct `eng` of the generated rows + gold, formal register, length-controlled)
is shared.  For each language: per-lang prompts split m_text 30% / hm 40% / ood 30%; hm switch language is drawn from the profile's pin list with the generator's
weights (first language x3, others x1.5; Pidgin: equal); the English length band follows what was generated for that language (brief languages: 1-3 sentences only).
Evaluation prompts are the English whose sha1 falls in the SFT val/test buckets (same rule as build_sft.split_of), never used for training prompts.
"""
from __future__ import annotations

from csw.paths import DATA_ROOT
import argparse
import collections
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = DATA_ROOT
from csw.datagen import gates                                   # noqa: E402
from csw.datagen.profiles import P, TARGETS, HOLD, BRIEF   # noqa: E402
from csw.sft import fmt                # noqa: E402
from csw.sft import rewards                # noqa: E402

STRONG = "swh yor hau ibo zul amh som orm xho afr bam sna tsn kin nya mlg sot wol lug ful".split()
MIX = (("m_text", 0.30), ("hm", 0.40), ("ood", 0.30))


def bucket(eng: str) -> str:
    h = int(hashlib.sha1(eng.strip().lower().encode()).hexdigest(), 16) % 100
    return "eval" if h < 4 else "train"


def load_english(rows_dir: str) -> dict[str, dict]:
    pool: dict[str, dict] = {}
    for f in sorted(glob.glob(str(Path(rows_dir) / "*.raw.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            e = " ".join((r.get("eng") or "").split())
            if len(e) < 20 or len(e) > 3200 or r.get("error"):
                continue
            k = e.lower()
            if k not in pool:
                sc = r.get("scenario") or {}
                pool[k] = {"english": e, "n_sent": gates.sentences(e), "genre": sc.get("genre"), "domain": sc.get("domain")}
    return pool


def load_fresh(path: str) -> dict[str, dict]:
    pool: dict[str, dict] = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            pool.setdefault(r["english"].lower(), {"english": r["english"], "n_sent": gates.sentences(r["english"]), "genre": r.get("register"), "domain": r.get("domain"),
                                                  "source": r.get("source")})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows"); ap.add_argument("--english"); ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=10000); ap.add_argument("--per-lang-strong", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pool = load_fresh(a.english) if a.english else load_english(a.rows)
    by_band = {"brief": [], "std": []}
    for v in pool.values():
        v["bucket"] = bucket(v["english"])
        (by_band["brief"] if v["n_sent"] <= 3 else by_band["std"]).append(v)
    langs = [l for l in TARGETS if l not in HOLD and l in fmt.NAMES]
    stats = collections.defaultdict(lambda: collections.Counter())
    rows = {"train": [], "eval": []}
    for lang in langs:
        rnd = random.Random(f"{a.seed}-{lang}")
        brief = lang in BRIEF or P[lang].tier == "C"
        cand = by_band["brief"] if brief else [v for v in pool.values() if v["n_sent"] >= 2]
        tr = [v for v in cand if v["bucket"] == "train"]; ev = [v for v in cand if v["bucket"] == "eval"]
        n = a.per_lang_strong if lang in STRONG else a.per_lang
        sw_list = [fmt.SWITCH_TAG[s] for s in P[lang].switch if s in fmt.SWITCH_TAG]
        sw_w = [1 if lang == "pcm" else (3 if i == 0 else 1.5) for i in range(len(sw_list))]
        for split, src, total in (("train", tr, n), ("eval", ev, max(60, n // 50))):
            rnd.shuffle(src)
            for task, share in MIX:
                k = int(round(total * share))
                # sample without replacement inside (language, task) as far as the pool allows, then cycle
                picks = [src[i % len(src)] for i in range(k)] if src else []
                rnd.shuffle(picks)
                for j, v in enumerate(picks):
                    sw = rnd.choices(sw_list, sw_w)[0] if task == "hm" else None
                    prompt = fmt.prompt_m_text(lang, v["english"]) if task == "m_text" else (fmt.prompt_hm(lang, sw, v["english"]) if task == "hm" else fmt.prompt_ood(lang, v["english"]))
                    rows[split].append({"id": f"{lang}-{task}-{split}-{j:05d}", "lang": lang, "task": task, "switch": sw, "prompt": prompt,
                                        "english": v["english"], "n_sent": v["n_sent"], "must_keep": rewards.must_keep(v["english"]), "genre": v["genre"], "domain": v["domain"], "source": v.get("source")})
                    stats[lang][f"{split}/{task}"] += 1
    for split in rows:
        random.Random(a.seed).shuffle(rows[split])
        with open(out / f"rl_prompts_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"english_pool": len(pool), "brief_pool": len(by_band["brief"]), "per_language": {l: dict(c) for l, c in stats.items()},
               "totals": {s: len(r) for s, r in rows.items()}}
    (out / "stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"english_pool": len(pool), "brief_pool": len(by_band["brief"]), "totals": summary["totals"]}))


if __name__ == "__main__":
    main()
