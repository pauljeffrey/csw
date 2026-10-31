"""Per-language generation budget for the CPU writer.  python -m csw.datagen.budgets  ->  budgets.json   (run where the Gemma tokenizer is available)
Tier HIGH (large web presence, the small model copes): max_new 350-512 tokens.  Tier LOW (mid/low resource, small model is unreliable): 50-150 tokens.
Inside a tier the budget follows how well Gemma writes the language (ratio_m = share of invented words relative to real text, measured earlier on the 31B in lang_support.json):
   q = clip((2.0 - ratio)/1.0, 0, 1);  HIGH: 350 + 162 q;  LOW: 50 + 100 q.
Target words = 0.80 x max_new / fertility, where fertility = Gemma tokens per word measured on that language's own seed text (African languages cost 2-4 tokens/word), so the document ends naturally before the cut-off."""
from csw.paths import DATA_ROOT
import json, os, sys, statistics as st, re
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import seeds as seedmod
from csw.datagen.profiles import P, TARGETS, HIGH, HOLD
from csw.datagen import cpu_gen
HIGH_T = list(HIGH) + ["pcm"]
tokpath = sorted(Path(os.environ.get("CSW_TOKENIZER_SNAPSHOTS", "hf/hub/models--RedHatAI--gemma-4-31B-it-FP8-dynamic/snapshots")).glob("*"))[0]   # NOTE: HF cache snapshot dir of the generator model
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(tokpath))
ls = json.loads((HERE / "lang_support.json").read_text()); out = {}
for l in TARGETS:
    if l in HOLD: continue
    texts = [s["text"] for s in seedmod.load(l) if s["track"] == "B" and cpu_gen.clean_seed(s["text"])][:300]
    if not texts: texts = [a["gloss"] for a in cpu_gen.anchors(l)][:300]
    words = sum(len(t.split()) for t in texts); toks = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
    f = toks / max(1, words); ratio = (ls.get(l) or {}).get("ratio_m") or 1.5; q = min(1, max(0, (2.0 - ratio) / 1.0))
    high = l in HIGH_T; mx = round(350 + 162 * q) if high else round(50 + 100 * q)
    out[l] = {"tier": "high" if high else "low", "fertility": round(f, 2), "ratio_m": ratio, "q": round(q, 2), "max_new": mx, "target_words": max(20, int(0.80 * mx / f))}
(HERE / "budgets.json").write_text(json.dumps(out, indent=1))
print(f"{'lang':5}{'tier':>5}{'tok/word':>9}{'ratio':>7}{'max_new':>8}{'words':>6}")
for l, v in sorted(out.items(), key=lambda kv: (kv[1]["tier"], -kv[1]["max_new"])): print(f"{l:5}{v['tier']:>5}{v['fertility']:>9}{v['ratio_m']:>7}{v['max_new']:>8}{v['target_words']:>6}")
