"""80k culture seed cards for the CPU writer - built offline (no GPU, no model): nothing here can slow the code-switching run.  python -m csw.datagen.build_culture_pool [--total 80000]
Two grounded sources, both already on disk:
  anchor  - a cultural concept (term + gloss + category) from anchors/<lang>.jsonl, e.g. "Mwaka Kogwa = a Shirazi New Year festival ..."  -> write about this concept in its community
  passage - a REAL native passage (explain/story/law/opinion seeds, cleaned) -> write a fresh document on the same subject (never copy it)
Each card gets a place (the language's dialect / city / community labels, rotated so every place is covered evenly) and is unique per language (normalised term / passage).
Quotas: high-resource languages up to HIGH_Q cards, low-resource up to LOW_Q, half anchor / half passage where both exist; the cards written by the 31B/CPU models in culture_seeds/ are kept on top.
Output: culture_pool/<lang>.jsonl"""
from csw.paths import DATA_ROOT
import json, re, sys, hashlib, random, collections
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import seeds as seedmod
from csw.datagen import cpu_gen
from csw.datagen.profiles import P, TARGETS, HOLD
B = json.loads((HERE / "budgets.json").read_text())
total = int(sys.argv[sys.argv.index("--total") + 1]) if "--total" in sys.argv else 80000
EN_LIKE = re.compile(r"^[a-z ]+$")
ANGLES = ["its meaning and where it comes from", "how people practise it today in everyday life", "one person's experience of it (a first-person account)", "what young people think about it now", "how it differs between neighbouring communities",
          "the part it plays in family and community life", "how it has changed over the last generations", "how a child learns about it", "the objects, food, music or places connected to it", "a problem or disagreement around it and how people settle it"]
GENRES = {"explain_teach": 3, "story": 3, "tradition_law": 3, "opinion": 2}        # chat/tweets excluded: little culture, much noise
out = Path(HERE / "culture_pool"); out.mkdir(exist_ok=True)
avail = {}
for l in B:
    A = {}
    for a in cpu_gen.anchors(l):
        k = re.sub(r"\W+", " ", a["term"].lower()).strip()
        if k and k not in A and a.get("gloss"): A[k] = a
    S = {}
    allb = [s for s in seedmod.load(l) if s["track"] == "B"]
    freq = collections.Counter(w for s in allb for w in set(cpu_gen.tk(s["text"])))              # words seen in >=3 seeds of this language: a passage made of rare words is probably another language
    def native(txt):
        ws = [w for w in cpu_gen.tk(txt) if len(w) > 2]
        return ws and sum(freq[w] >= 3 for w in ws) / len(ws) >= 0.55
    for s in (allb if B[l]["tier"] == "high" else []):                       # low-resource seed corpora are mostly parallel Bible/web pairs with a second language mixed in -> anchors only for them
        if s["genre"] not in GENRES or not cpu_gen.clean_seed(s["text"]) or not 25 <= len(s["text"].split()) <= 110: continue
        if re.search(r"\+\d|\d\*|\*\+|^\d+ ", s["text"]) or sum(c.isdigit() for c in s["text"]) / len(s["text"]) > 0.01 or not native(s["text"]): continue          # verse numbers, mixed-language text
        k = hashlib.sha1(re.sub(r"\W+", " ", s["text"].lower()).strip()[:160].encode()).hexdigest()
        if k not in S: S[k] = s
    avail[l] = (list(A.values()), list(S.values()))
tier = {l: B[l]["tier"] for l in B}
w = {l: (3.0 if tier[l] == "high" else 1.0) for l in B}                      # high-resource languages get 3x the cards of a low-resource one
llm = {l: sum(1 for _ in open(HERE / "culture_seeds" / f"{l}.jsonl", encoding="utf-8")) if (HERE / "culture_seeds" / f"{l}.jsonl").exists() else 0 for l in B}
unit = total / sum(w.values()); quota = {l: int(unit * w[l]) for l in B}
summary = {}
for l in sorted(B):
    A, S = avail[l]; p = P[l]; places = list(p.dialects) + [r for r in p.persona_regions if r not in p.dialects]
    rnd = random.Random(f"pool-{l}"); rnd.shuffle(A); rnd.shuffle(S)
    q = max(0, quota[l] - llm[l]); na = min(len(A), q // 2); ns = min(len(S), q - na); na = min(len(A), q - ns)
    cards = []
    for i, a in enumerate(A[:na]):
        cards.append({"id": hashlib.sha1(("a" + l + a["term"]).encode()).hexdigest()[:12], "type": "anchor", "lang": l, "country": None, "region": places[i % len(places)], "community": None, "aspect": a.get("cat"), "title": a["term"], "description": a["gloss"], "terms": [a["term"]] if a.get("untranslatable") else []})
    for i, s in enumerate(S[:ns]):
        cards.append({"id": hashlib.sha1(("p" + l + s["seed_id"]).encode()).hexdigest()[:12], "type": "passage", "lang": l, "country": None, "region": places[(i + 1) % len(places)], "community": None, "aspect": s["genre"], "title": "subject of a real passage", "description": s["text"], "terms": []})
    need = max(0, quota[l] - llm[l] - len(cards))                      # beyond the unique concepts/passages: the SAME concept seen from another place and another angle (a different document)
    if need and A:
        for j in range(need):
            a = A[j % len(A)]; rnd_i = j // len(A) + 1
            place = places[(j + rnd_i * 3 + 1) % len(places)]; ang = ANGLES[(j * 7 + rnd_i) % len(ANGLES)]
            cards.append({"id": hashlib.sha1(("v" + l + a["term"] + place + ang).encode()).hexdigest()[:12], "type": "anchor", "lang": l, "country": None, "region": place, "community": None, "aspect": a.get("cat"), "title": a["term"], "description": a["gloss"],
                          "terms": [a["term"]] if a.get("untranslatable") else [], "angle": ang})
    seen = set(); uniq = []
    for c in cards:
        if c["id"] not in seen: seen.add(c["id"]); uniq.append(c)
    cards = uniq
    rnd.shuffle(cards)
    (out / f"{l}.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cards) + ("\n" if cards else ""), encoding="utf-8")
    summary[l] = (len(cards), llm[l], len(A), len(S), quota[l])
tot = sum(v[0] + v[1] for v in summary.values())
print(f"{'lang':5}{'tier':>5}{'cards':>7}{'llm':>5}{'anchors avail':>14}{'passages avail':>15}{'quota':>7}")
for l, v in sorted(summary.items(), key=lambda kv: (tier[kv[0]], -kv[1][0])): print(f"{l:5}{tier[l]:>5}{v[0]:>7}{v[1]:>5}{v[2]:>14}{v[3]:>15}{v[4]:>7}")
print("TOTAL cards (pool + LLM cards):", tot)
