"""English DOMAIN seed pool (money / law / health / tech / family / politics / school / sport / work / stories / chat), streamed from labelled
Hub datasets, filtered for simplicity (Flesch-Kincaid, lowest first), length and cleanliness.  Run on an instance (streams, no big download):
    python -m csw.datagen.build_en_domain OUT.jsonl [per_domain=300]"""
import hashlib, html, json, os, random, re, sys, statistics as st
from datasets import load_dataset
out, per = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300
tok = os.environ.get("HF_TOKEN")
D = {"money": "money and informal finance", "health": "health and body", "tech": "technology and phones", "law": "law, disputes and custom", "politics": "politics and community",
     "family": "family and marriage", "school": "school and learning", "sport": "sports and entertainment", "work": "work and hustle", "travel": "travel and transport",
     "farm": "farming and food", "faith": "religion and faith", "weather": "weather, land and animals", "identity": "identity, migration and generation gap"}
KW = {"money": "bank loan saving savings interest credit debt invest salary price budget mortgage tax cost pay payment cash income insurance", "law": "court judge lawyer contract police sue legal law landlord tenant evict custody inheritance fine arrest",
      "health": "doctor hospital medicine fever pain disease symptom clinic nurse pregnant blood infection vaccine diet headache", "tech": "phone app internet computer network wifi data software password email website laptop battery online",
      "farm": "farm crop harvest soil seed cattle goat maize rice garden fertilizer", "travel": "bus train road car taxi flight airport driver traffic journey", "school": "school teacher student exam class university homework college lesson",
      "work": "job boss office salary interview employer colleague business shop trade customer", "family": "husband wife mother father child baby marriage wedding sister brother family", "faith": "church mosque pray prayer god pastor imam", "weather": "rain flood drought weather storm wind heat"}
def dom_kw(t):
    w = set(re.findall(r"[a-z]+", t.lower())); best = max(KW, key=lambda k: len(w & set(KW[k].split())))
    return best if len(w & set(KW[best].split())) >= 2 else None
def syll(w):
    c = len(re.findall(r"[aeiouy]+", w.lower())); return max(1, c - (1 if w.lower().endswith("e") and not w.lower().endswith(("le", "ee")) and c > 1 else 0))
def fk(t):
    s = max(1, len(re.findall(r"[.!?]+", t))); ws = re.findall(r"[A-Za-z']+", t); n = max(1, len(ws)); return 0.39 * n / s + 11.8 * sum(syll(x) for x in ws) / n - 15.59
BAD = re.compile(r"(https?://|www\.|@\w+\.|<[a-z/]+>|&[a-z]+;|\bfuck|\bshit\b|\bsex\b|\bporn|\bnude\b|\bkill myself|\bsuicid|\brape\b|\bnazi)", re.I)
def clean(t):
    t = html.unescape(re.sub(r"<[^>]+>", " ", t or "")); return re.sub(r"\s+", " ", t).strip()
def ok(t):
    wc = len(t.split())
    return 20 <= wc <= 110 and not BAD.search(t) and sum(c.isascii() for c in t) / max(1, len(t)) > 0.97 and len(re.findall(r"[.!?]", t)) >= 2
pool = {}   # (domain, genre, src, license) -> list of (fk, text)
def add(dkey, genre, src, lic, t):
    t = clean(t)
    if dkey and ok(t): pool.setdefault((dkey, genre, src, lic), []).append((fk(t), t))
def stream(name, split="train", cfg=None, limit=20000, **kw):
    ds = load_dataset(name, cfg, split=split, streaming=True, token=tok, **kw)
    for i, r in enumerate(ds):
        if i >= limit: break
        yield r
log = lambda m: print(m, flush=True)
# yahoo answers topics (real people's questions, labelled)
Y = {2: "health", 4: "tech", 6: "money", 9: "politics", 8: "family", 3: "school", 5: "sport"}
for r in stream("community-datasets/yahoo_answers_topics", limit=90000):
    if r["topic"] in Y: add(Y[r["topic"]], "chat", "yahoo_answers_topics", "yahoo-terms (seed use only)", f"{r['question_title']}. {r['question_content']}")
log("yahoo done")
for r in stream("keivalya/MedQuad-MedicalQnADataset", limit=16000): add("health", "explain_teach", "MedQuad", "cc0/nih", f"{r['Question']} {' '.join(clean(r['Answer']).split('. ')[:2])}")
log("medquad done")
for r in stream("FinGPT/fingpt-fiqa_qa", limit=17000): add("money", "chat", "fiqa", "cc-by-sa/stackexchange", r["input"])
log("fiqa done")
for r in stream("jonathanli/law-stack-exchange", limit=20000): add("law", "chat", "law.stackexchange", "cc-by-sa", f"{r['title']}. {r['body']}")
log("law done")
SHP = {"legaladvice": "law", "askdocs": "health", "askhr": "work", "askcarguys": "travel"}
for r in stream("stanfordnlp/SHP", limit=120000):
    d = SHP.get(r["domain"].split("_")[0]) if r.get("domain") else None
    if d: add(d, "chat", "SHP/" + r["domain"].split("_")[0], "research-only", r["history"])
log("shp done")
for r in stream("databricks/databricks-dolly-15k", limit=15000):
    if r["category"] in ("general_qa", "open_qa", "brainstorming"): add(dom_kw(r["instruction"] + " " + r["response"]), "explain_teach", "dolly/" + r["category"], "cc-by-sa", f"{r['instruction']} {r['response']}")
log("dolly done")
for r in stream("allenai/soda", limit=30000):
    t = " ".join(f"{s}: {u}" for s, u in zip(r["speakers"][:6], r["dialogue"][:6])); add(dom_kw(t) or "family", "chat", "soda", "cc-by-4.0", t)
log("soda done")
g = 0
for r in stream("sedthh/gutenberg_english", limit=400):
    for para in re.split(r"\n\s*\n", r["TEXT"]):
        p = clean(para)
        if ok(p) and fk(p) <= 8 and '"' in p or ok(p) and fk(p) <= 6: add("identity", "story", "gutenberg", "public-domain", p); g += 1
        if g > 4000: break
    if g > 4000: break
log("gutenberg done")
rows = []
for (dkey, genre, src, lic), items in pool.items():
    items.sort(key=lambda x: x[0]); k = per if src not in ("gutenberg", "soda") else per * 2
    for f, t in items[:k]: rows.append({"seed_id": hashlib.md5(t.encode()).hexdigest()[:12], "lang": "_en", "track": "C", "genre": genre, "domain": D[dkey], "text": t, "src": src, "license": lic, "fk_grade": round(f, 2), "words": len(t.split())})
random.Random(5).shuffle(rows)
with open(out, "w", encoding="utf-8") as fo:
    for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
by = {}
for r in rows: by.setdefault(r["domain"], []).append(r["fk_grade"])
log(f"{len(rows)} seeds -> {out}")
for d, v in sorted(by.items(), key=lambda x: -len(x[1])): log(f"  {d:34} n={len(v):4}  FK median {st.median(v):.1f}  max {max(v):.1f}")
