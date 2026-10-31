"""Build small raw files for the back-translation check: GOLD rows (known good, the noise baseline) vs GENERATED rows (pilot + main).
python -m csw.datagen.bt_sets [--n 8]  ->  out_bt_gold/<lang>.raw.jsonl, out_bt_gen/<lang>.raw.jsonl"""
from csw.paths import DATA_ROOT
import json, sys, collections
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import cs_prompts as pr
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8
LANGS = ["yor", "hau", "ibo", "swh", "zul", "som", "amh", "twi", "lin", "kin", "tso", "kon"]
def w(d, lang, rows):
    Path(HERE / d).mkdir(exist_ok=True)
    (HERE / d / f"{lang}.raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
aug = pr.load_aug_gold()
for l in LANGS:
    g = [{"id": f"{l}-gold-{i}", "lang": l, "eng": r["eng"], "m_text": r["m_text"], "scenario": {"dialect": r.get("dialect") or "", "length": "short"}} for i, r in enumerate(aug.get(l, [])[:n])]
    if g: w("out_bt_gold", l, g)
gen = collections.defaultdict(list)
for f in [HERE / "out_pilot" / "pilot_T0.6.jsonl"]:
    if f.exists():
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                o = json.loads(line)
                if o.get("ok"): r = o["row"]; gen[r["lang"]].append({"id": r["id"], "lang": r["lang"], "eng": r["eng"], "m_text": r["m_text"], "scenario": {"dialect": r["scenario"].get("dialect", ""), "length": r["scenario"]["length"]}})
for f in (HERE / "out_main").glob("*.raw.jsonl"):
    for line in f.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            if r.get("m_text") and r.get("eng") and r["lang"] in LANGS: gen[r["lang"]].append({"id": r["id"], "lang": r["lang"], "eng": r["eng"], "m_text": r["m_text"], "scenario": {"dialect": r["scenario"].get("dialect", ""), "length": r["scenario"]["length"]}})
for l, rows in gen.items():
    seen = set(); u = [r for r in rows if not (r["id"] in seen or seen.add(r["id"]))][: n * 2]
    w("out_bt_gen", l, u)
print({d: sum(1 for _ in (HERE / d).glob("*.raw.jsonl")) for d in ("out_bt_gold", "out_bt_gen")})
