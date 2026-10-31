"""Pick the sampling configuration empirically on the real server: the SAME 60 task cards (12 languages x 5 rows incl. long; see pilot_select.py) are generated under
4 configs, scored with the rule-based scorer, and the best config is written to gen_config.json (read by vast/gen_rounds.sh).
    CS_LOCAL_URL=http://127.0.0.1:8000/v1 python -m csw.datagen.pilot_sampling [--model google/gemma-4-31b-it] [--rows 5]
Choice: highest mean rule_score minus 0.5 x hard-failure rate minus 0.5 x loop rate; ties -> lower model difficulty. The config list is in CONFIGS."""
from csw.paths import DATA_ROOT
import json, os, sys, statistics as st, collections as C
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = DATA_ROOT
os.environ.setdefault("CS_OUT", "out_pilot")
from csw.datagen import cs_pipeline as pl
from csw.datagen import cs_prompts as pr
from csw.judging import score_rows as sr
from csw.datagen.providers import Request
from csw.datagen.profiles import TARGETS, LENGTHS
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "google/gemma-4-31b-it"
n_rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 5
CONFIGS = [{"name": "T0.4", "temperature": 0.4, "rep": 1.05, "top_p": 0.90}, {"name": "T0.6", "temperature": 0.6, "rep": 1.05, "top_p": 0.95},
           {"name": "T0.8", "temperature": 0.8, "rep": 1.05, "top_p": 0.95}, {"name": "T0.6-norep", "temperature": 0.6, "rep": 1.0, "top_p": 0.95}]
pl.PROVIDER = "local"; pl.GOLD = pr.load_gold()
from csw.datagen import pilot_select
cards = pilot_select.select(n_rows)        # the same cards as exemplars/pilot_cards.jsonl (manually reviewed)
print(len(cards), "task cards,", len(CONFIGS), "configs =", len(cards) * len(CONFIGS), "generations", flush=True)
res = {}
only = sys.argv[sys.argv.index("--configs") + 1].split(",") if "--configs" in sys.argv else None
OUT = HERE / "out_pilot"; OUT.mkdir(exist_ok=True)
for cfg in [c for c in CONFIGS if not only or c["name"] in only]:
    os.environ["CS_REP_PEN"] = str(cfg["rep"]); pv = pl.prov(model, 24)
    def one(c):
        r = pv.complete(Request(c["id"], pr.gen_prompt(pl.P[c["lang"]], c, pl.GOLD), max_tokens=LENGTHS[c["length"]][2], temperature=cfg["temperature"], top_p=cfg["top_p"], response_format={"type": "json_object"}))
        d = pr.parse_json(r.text) if r.ok else None
        if not d or not all(d.get(k) for k in ("eng", "m_text", "hm", "ood")): return {"ok": False, "id": c["id"], "lang": c["lang"], "error": r.error or "bad_json", "raw": (r.text or "")[:600]}
        row = {"id": c["id"], "lang": c["lang"], "scenario": {"switch": c["switch"], "length": c["length"], "n_sent": c["n_sent"], "length_spec": "v2", "genre": c["genre"], "track": c["track"], "dialect": c["dialect"]}, "seed_id": c["seed_id"], **{k: d.get(k) for k in ("eng_initial", "eng", "m_text", "hm", "ood", "ood_relation", "devices", "conf", "flags", "self_check", "difficulty")}}
        s = sr.score_row(row); g = sr.gates.check(dict(row), pl.P[c["lang"]])
        en = sr.gates.sentences(row["eng"]); want = c["n_sent"]; tol = 1 if want <= 6 else (2 if want <= 12 else 3)
        return {"ok": True, "score": s["rule_score"], "hard": any(f.startswith("gate:") for f in s["flags"]), "loop": s["components"]["repetition"] == 0.0, "diff": row["difficulty"] if isinstance(row["difficulty"], (int, float)) else None,
                "len_ok": abs(en - want) <= tol, "len_exact": en == want, "en_sent": en, "want": want, "hm_en": g["metrics"].get("en_hm"), "row": row, "scores": s, "usage": [r.prompt_tokens, r.completion_tokens]}
    with ThreadPoolExecutor(24) as ex: out = list(ex.map(one, cards))
    ok = [o for o in out if o["ok"]]
    with open(OUT / f"pilot_{cfg['name']}.jsonl", "w", encoding="utf-8") as fo:
        for o in out: fo.write(json.dumps(o, ensure_ascii=False) + "\n")
    res[cfg["name"]] = {"valid_json_rate": round(len(ok) / len(out), 3), "mean_rule_score": round(st.mean(o["score"] for o in ok), 3) if ok else 0, "hard_fail_rate": round(sum(o["hard"] for o in ok) / max(1, len(ok)), 3),
                        "loop_rate": round(sum(o["loop"] for o in ok) / max(1, len(ok)), 3), "length_ok_rate": round(sum(o["len_ok"] for o in ok) / max(1, len(ok)), 3), "length_exact_rate": round(sum(o["len_exact"] for o in ok) / max(1, len(ok)), 3),
                        "mean_completion_tokens": round(st.mean(o["usage"][1] for o in ok)) if ok else 0, "mean_difficulty": round(st.mean(o["diff"] for o in ok if o["diff"] is not None), 2) if any(o["diff"] is not None for o in ok) else None, **cfg}
    print(cfg["name"], {k: v for k, v in res[cfg["name"]].items() if k not in ("name", "temperature", "rep", "top_p")}, flush=True)
key = lambda r: (r["valid_json_rate"] * r["mean_rule_score"] - 0.5 * r["hard_fail_rate"] - 0.5 * r["loop_rate"], -(r["mean_difficulty"] or 0))
best = max(res.values(), key=key)
json.dump({"chosen": {k: best[k] for k in ("name", "temperature", "rep", "top_p")}, "top_k": 40, "evidence": res}, open(HERE / "gen_config.json", "w"), indent=1)
print("CHOSEN:", best["name"], "-> gen_config.json")
