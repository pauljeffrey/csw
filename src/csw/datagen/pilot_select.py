"""The 60 pilot task cards (12 languages x 5), shared by pilot_sampling.py and export. Held languages (profiles.HOLD) are excluded; banded languages get
2 short + 2 medium + 1 long card so the length bins are tested too; brief languages get 5 brief cards; track D cards need the English pool, so they are skipped.
python -m csw.datagen.pilot_select  ->  exemplars/pilot_cards.jsonl   (full system + user prompt per card, for review)"""
from csw.paths import DATA_ROOT
import json, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import cs_pipeline as pl
from csw.datagen import cs_prompts as pr
from csw.datagen.profiles import TARGETS, LENGTHS, HOLD, BRIEF

LANGS = ["yor", "hau", "ibo", "swh", "zul", "som", "amh", "twi", "lin", "kin", "tso", "kon"]     # strong, mid and brief (weakest-allowed) languages; no held language
assert not set(LANGS) & set(HOLD)


def select(n_rows: int = 5) -> list[dict]:
    cards = []
    for lang in LANGS:
        plan = [c for c in pl.make_plan(lang, TARGETS[lang]) if c["track"] != "D"]
        if lang in BRIEF or pl.P[lang].tier == "C":
            cards += plan[:n_rows]
        else:
            want = {"short": 2, "medium": 2, "long": 1}
            for ln, k in want.items():
                cards += [c for c in plan if c["length"] == ln][:k]
    return cards


def main():
    gold = pr.load_gold(); out = []
    for c in select():
        m = pr.gen_prompt(pl.P[c["lang"]], c, gold)
        out.append({"id": c["id"], "lang": c["lang"], "track": c["track"], "length": c["length"], "n_sent": c["n_sent"], "genre": c["genre"], "switch": c["switch"], "dialect": c["dialect"],
                    "max_tokens": LENGTHS[c["length"]][2], "system_variant": pr.variant_for(c["lang"], c["id"]), "system": m[0]["content"], "user": m[1]["content"]})
    p = HERE / "exemplars" / "pilot_cards.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    import collections as C
    print(len(out), "cards ->", p, dict(C.Counter(r["length"] for r in out)), dict(C.Counter(r["track"] for r in out)))


if __name__ == "__main__":
    main()
