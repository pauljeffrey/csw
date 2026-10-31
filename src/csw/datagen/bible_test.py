"""Can Gemma translate Bible-style English into Efik / Urhobo?  Zero-shot vs 5-shot with retrieved REAL parallel verses; chrF vs the reference.
python -m csw.datagen.bible_test [n_test=24]"""
from csw.paths import DATA_ROOT
import json, os, re, sys, random, statistics as st, time
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import seeds
from csw.datagen import cs_pipeline as pl
from csw.datagen.providers import Request, get_provider
from concurrent.futures import ThreadPoolExecutor
n_test = int(sys.argv[1]) if len(sys.argv) > 1 else 24
DS = {"efi": ("michsethowusu/english-efik_sentence-pairs_mt560", "efi", "Efik"), "urh": ("michsethowusu/english-urhobo_sentence-pairs_mt560", "urh", "Urhobo")}
STOP = set("the a an and of to in on at for with from by is are was were be been it its i you he she we they me him her us them my your his our their this that these those shall will would should unto not no".split())
os.environ["CS_REASONING"] = "none"; pl.PROVIDER = "openrouter"
pv = pl.prov("google/gemma-4-31b-it", 6)
W = lambda t: [w for w in re.findall(r"[a-z']+", t.lower()) if w not in STOP]
res = {}
for lang, (ds, col, name) in DS.items():
    pool = []
    for attempt in range(3):
        try:
            rows = seeds.fetch_rows(ds, 300, seed=attempt + 7); pool = [(r["eng"].strip(), r[col].strip()) for r in rows if 8 <= len(r["eng"].split()) <= 35 and r.get(col)]; break
        except Exception as e:
            time.sleep(20)
    if len(pool) < n_test + 20: print(lang, "pool too small", len(pool)); continue
    random.Random(3).shuffle(pool); test, shots = pool[:n_test], pool[n_test:]
    def shots_for(eng):
        q = set(W(eng)); sc = sorted(shots, key=lambda p: -len(q & set(W(p[0]))) / (1 + len(set(W(p[0]))) ** 0.5))
        return sc[:5]
    SYS = f"You are a professional translator into {name}. Translate the English into {name}, keeping the meaning, names and style of Bible/religious text. Use the standard written orthography of {name} with all diacritics and special letters. Output ONLY the {name} translation."
    def run(item, k):
        eng, ref = item
        ex = shots_for(eng) if k else []
        msgs = [{"role": "system", "content": SYS}]
        for e, t in ex: msgs += [{"role": "user", "content": e}, {"role": "assistant", "content": t}]
        msgs.append({"role": "user", "content": eng})
        r = pv.complete(Request("bt", msgs, max_tokens=600, temperature=0.0))
        return (r.text or "").strip()
    with ThreadPoolExecutor(6) as ex:
        z = list(ex.map(lambda it: run(it, 0), test)); f = list(ex.map(lambda it: run(it, 5), test))
    cz = [pl.chrf(h, ref[1]) for h, ref in zip(z, test)]; cf = [pl.chrf(h, ref[1]) for h, ref in zip(f, test)]
    # control: chrF between two unrelated references (what 'plausible-looking but wrong' scores)
    ctrl = [pl.chrf(test[i][1], test[(i + 1) % len(test)][1]) for i in range(len(test))]
    res[lang] = {"n": len(test), "chrF_zero_shot": round(st.mean(cz), 1), "chrF_5shot_real_pairs": round(st.mean(cf), 1), "chrF_unrelated_sentence_control": round(st.mean(ctrl), 1),
                 "examples": [{"eng": test[i][0], "ref": test[i][1], "zero": z[i], "five": f[i]} for i in range(3)]}
    print(lang, {k: v for k, v in res[lang].items() if k != "examples"}, flush=True)
json.dump(res, open(HERE / "bible_test.json", "w"), ensure_ascii=False, indent=1)
for lang, v in res.items():
    print(f"\n=== {lang} example")
    e = v["examples"][0]
    for k in ("eng", "ref", "zero", "five"): print(f"  {k:5}: {e[k][:240]}")
