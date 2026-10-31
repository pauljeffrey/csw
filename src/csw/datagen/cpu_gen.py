"""Monolingual African-culture text for pretraining, written by a small Gemma model on the box's idle CPU (llama.cpp server).
python -m csw.datagen.cpu_gen eval --url http://127.0.0.1:8001/v1 --model NAME --tag 26b [--per-lang 6]            speed + word-validity test (6 high-resource + 3 low-resource languages)
python -m csw.datagen.cpu_gen run  --url ... --model NAME --tag 26b --tier high|low [--langs ...] [--conc 8] [--hours H]   open-ended generation -> out_cpu/<lang>.jsonl (kept docs) + <lang>.all.jsonl
Budgets: budgets.json (budgets.py) gives each language its max_new tokens (high tier 350-512, low tier 50-150, scaled by how well Gemma writes the language) and a target word count from the measured
tokens-per-word of that language. High and low tiers run as separate phases so the requests in flight have similar lengths (no long-tail stalls).
Topics: culture_seeds/<lang>.jsonl (culture_seeds.py): distinct, factual aspects of different countries / regions / communities / dialect areas, each card used once. A real native passage gives the style."""
from csw.paths import DATA_ROOT
import json, os, random, re, sys, time, collections, statistics as st, hashlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = DATA_ROOT
from csw.datagen import seeds as seedmod
from csw.datagen.profiles import P
W = re.compile(r"[^\W\d_]+", re.U); tk = lambda t: [w.lower() for w in W.findall(t)]
EN = set("the a an and or but of to in on at for with from by is are was were be been being have has had do does did not this that these those it its i you he she we they what which who when where why how if then than so very just also can could will would shall should may might must about into over after before again more most some any all each every too only there here as because while".split())
DOC_TYPES = ["an encyclopedic article that explains", "a short story (folktale or realistic anecdote) set around", "a magazine-style feature that describes", "a first-person recollection from an elder or a parent about",
             "a practical explainer (how people do it) about", "a radio-talk style essay about", "a school textbook passage on", "a thoughtful essay about"]
BAD = re.compile(r"Translation of|individual language|Text in |Translation to|\bEnglish\b|https?://|\d{4,}", re.I)
SENT_END = re.compile(r"[.!?…።؟]+[\"'”’)]*\s")


def clean_seed(s):                                       # corpus seeds sometimes carry parallel-text scaffolding or another language's translation
    d = sum(c.isdigit() for c in s) / max(1, len(s))
    return not BAD.search(s) and d < 0.03


def vocab(lang):
    seeds = [s["text"] for s in seedmod.load(lang) if s["track"] == "B" and clean_seed(s["text"])]
    random.Random(1).shuffle(seeds); k = max(10, len(seeds) // 10)
    held, train = seeds[:k], seeds[k:]
    return set(w for t in train for w in tk(t)), held, train


def oov(ws, v):
    ws = [w for w in ws if len(w) > 2]
    return sum(w not in v for w in ws) / max(1, len(ws))


def anchors(lang):
    p = HERE / "anchors" / f"{lang}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()] if p.exists() else []


def budgets():
    return json.loads((HERE / "budgets.json").read_text())


def cards(lang):
    out = []
    for d in ("culture_seeds", "culture_pool"):                 # culture_seeds = cards written by Gemma-4 (places/aspects); culture_pool = offline anchor + real-passage cards (build_culture_pool.py)
        p = HERE / d / f"{lang}.jsonl"
        if p.exists(): out += [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]
    return out


def build_prompt(lang, rnd, card, B):
    p = P[lang]; b = B[lang]; words = b["target_words"]; dt = rnd.choice(DOC_TYPES)
    seeds = [s for s in seedmod.load(lang) if s["track"] == "B" and 20 <= len(s["text"].split()) <= (60 if b["tier"] == "low" else 110) and clean_seed(s["text"])]
    seed = rnd.choice(seeds)["text"] if seeds else ""
    size = f"about {words} words (a single short paragraph)" if b["tier"] == "low" else f"about {words} words"
    sysm = f"You are an expert native {p.name} writer and editor. You write clear, natural, correct {p.name} the way an educated native speaker writes it, and you never invent words, customs or proverbs: when unsure of a word you use a simpler word you are certain of."
    where = ", ".join(x for x in (card.get("community"), card.get("region"), card.get("country")) if x)
    if card.get("type") == "passage":                          # subject taken from a real native passage: the document must be new, not a paraphrase
        subj = f"the same SUBJECT as the real {p.name} passage below (identify what it is about and write a fresh, factual document on it - do not copy or paraphrase its sentences):\n\"\"\"{card.get('description')}\"\"\"\n"
        seed = ""
    else:
        subj = f"SUBJECT ({card.get('aspect')}): {card.get('title')} - {card.get('description')}\n" + (f"ANGLE: {card['angle']}\n" if card.get("angle") else "")
    head = f"Write a fresh, factual document (an article, essay, story or explainer - your choice) in {p.name}, {size}, on " if card.get("type") == "passage" else f"Write {dt} this subject, in {p.name}, {size}:\n"
    user = (head +
            f"{subj}PLACE / COMMUNITY: {where}\n"
            f"RULES: write ENTIRELY in {p.name} - no English words except proper names, no translation, no title, no notes. Use the standard orthography including every diacritic / special letter ({p.orthography}). "
            f"Stay with the facts given above; add only concrete details you are sure of and never invent names or customs. Do not copy the passage below.\n"
            + (f"Local words you may use if they fit: {', '.join(card['terms'][:4])}.\n" if card.get("terms") else "")
            + (f"A real {p.name} passage, ONLY as a guide to spelling, vocabulary and style:\n\"\"\"{seed}\"\"\"\n" if seed else "") + "Output only the document text.")
    return {"system": sysm, "user": user, "topic": card.get("title"), "aspect": card.get("aspect"), "place": where, "doc_type": dt, "target_words": words, "max_new": b["max_new"], "card_id": card.get("id")}


def metrics(text, V, real_oov):
    ws = tk(text); n = len(ws)
    g3 = [tuple(ws[i:i + 3]) for i in range(max(0, n - 2))]
    o = oov(ws, V)
    return {"words": n, "oov": round(o, 3), "oov_ratio": round(o / max(real_oov, 1e-3), 2), "en_share": round(sum(w in EN for w in ws) / max(1, n), 3), "distinct3": round(len(set(g3)) / max(1, len(g3)), 3)}


def keep(m, target, thr_ratio, lang=""):
    en_ok = lang == "afr" or m["en_share"] < 0.04                  # Afrikaans shares function words (is, as, so ...) with English: that check does not apply
    return m["words"] >= max(15, int(0.5 * target)) and en_ok and m["distinct3"] > 0.80 and m["oov_ratio"] <= thr_ratio


def trim_to_sentence(text):                              # a cut-off at max_new: drop the unfinished last sentence
    ends = [m.end() for m in SENT_END.finditer(text + " ")]
    return text[:ends[-1]].strip() if ends else text.strip()


def call(url, model, sysm, user, max_tokens, temp=0.7):
    import urllib.request
    body = {"model": model, "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": temp, "top_p": 0.95, "repeat_penalty": 1.05}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); ch = d["choices"][0]
    text = ch["message"].get("content") or ""
    if ch.get("finish_reason") == "length": text = trim_to_sentence(text)
    return text, (d.get("usage") or {}).get("completion_tokens", 0), time.time() - t0, ch.get("finish_reason")


def main():
    a = sys.argv; mode = a[1]; opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    url, model, tag = opt("--url"), opt("--model"), opt("--tag", "x"); conc = int(opt("--conc", 8)); B = budgets()
    if mode == "eval":
        per = int(opt("--per-lang", 6)); jobs = []; Vs = {}
        for l in ["swh", "hau", "yor", "amh", "zul", "afr", "tso", "tum", "luo"]:
            V, held, _ = vocab(l); Vs[l] = (V, oov([w for t in held for w in tk(t)], V)); rnd = random.Random(f"eval-{l}"); cs = cards(l); rnd.shuffle(cs)
            jobs += [(l, i, build_prompt(l, rnd, cs[i], B)) for i in range(min(per, len(cs)))]
        jobs.sort(key=lambda j: (B[j[0]]["tier"] != "high", j[0]))          # high tier first, then low tier: similar lengths run together
        Path(HERE / "out_cpu_eval").mkdir(exist_ok=True); fo = open(HERE / "out_cpu_eval" / f"{tag}.jsonl", "w", encoding="utf-8"); lk = threading.Lock(); res = []
        def one(j):
            l, i, pr = j; V, real = Vs[l]
            try: text, ct, sec, fin = call(url, model, pr["system"], pr["user"], pr["max_new"])
            except Exception as e: return None
            m = metrics(text, V, real)
            r = {"lang": l, "i": i, "model": model, "tier": B[l]["tier"], "text": text, "completion_tokens": ct, "sec": round(sec, 1), "finish": fin, "target_words": pr["target_words"], "max_new": pr["max_new"], **m, "keep": keep(m, pr["target_words"], 2.0, l), "topic": pr["topic"], "place": pr["place"]}
            with lk: fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush(); res.append(r)
        t0 = time.time()
        with ThreadPoolExecutor(conc) as ex: list(ex.map(one, jobs))
        wall = time.time() - t0; ct = sum(r["completion_tokens"] for r in res); by = collections.defaultdict(list)
        for r in res: by[r["lang"]].append(r)
        summ = {"tag": tag, "model": model, "docs": len(res), "wall_s": round(wall), "completion_tokens": ct, "tok_per_s": round(ct / wall, 1), "docs_per_hour": round(len(res) / wall * 3600), "kept": sum(r["keep"] for r in res),
                "mean_words": round(st.mean(r["words"] for r in res)), "mean_oov_ratio": round(st.mean(r["oov_ratio"] for r in res), 2), "mean_en_share": round(st.mean(r["en_share"] for r in res), 3),
                "mean_distinct3": round(st.mean(r["distinct3"] for r in res), 3), "truncated": sum(r["finish"] == "length" for r in res),
                "per_lang": {l: {"oov_ratio": round(st.mean(r["oov_ratio"] for r in v), 2), "words": round(st.mean(r["words"] for r in v)), "kept": sum(r["keep"] for r in v), "n": len(v)} for l, v in by.items()}}
        (HERE / "out_cpu_eval" / f"{tag}.summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ), flush=True)
    elif mode == "run":
        tier = opt("--tier", "high"); langs = []
        if "--langs" in a:
            i = a.index("--langs") + 1
            while i < len(a) and not a[i].startswith("--"): langs.append(a[i]); i += 1
        else: langs = [l for l, b in B.items() if b["tier"] == tier and cards(l)]
        thr = json.loads(opt("--thr", "{}")); deadline = time.time() + float(opt("--hours", 24)) * 3600
        Path(HERE / "out_cpu").mkdir(exist_ok=True); lk = threading.Lock(); Vs = {}; queues = {}; kept = collections.Counter()
        for l in langs:
            V, held, _ = vocab(l); Vs[l] = (V, oov([w for t in held for w in tk(t)], V)); cs = cards(l); random.Random(f"run-{l}").shuffle(cs); done = set()
            f = HERE / "out_cpu" / f"{l}.all.jsonl"
            if f.exists():
                for line in f.read_text(encoding="utf-8").split("\n"):
                    if line.strip(): r = json.loads(line); done.add(r.get("card_id")); kept[l] += bool(r.get("keep"))
            queues[l] = [c for c in cs if c["id"] not in done]
        rnd = random.Random(f"{tag}-{time.time()}")
        # Language choice (important languages first; nothing is discarded, so fairness is over documents WRITTEN, not documents kept - choosing by
        # 'kept' let languages whose documents the filter always rejects (the Pidgins) hog every slot).
        P1 = set("swh yor hau ibo zul amh som orm xho afr bam sna tsn kin nya mlg sot wol lug ful pcm".split())      # the 20 strongest languages + Nigerian Pidgin
        quota = lambda x: 150 if x in P1 else 100                    # first fill every language to its quota, important ones first, then continue 2:1 in favour of P1
        att = collections.Counter({l: len([c for c in cards(l)]) - len(queues[l]) for l in langs})      # cards already written (queues hold only the unwritten ones)
        def next_job():
            with lk:
                cand = [l for l in langs if queues[l]]
                if not cand: return None
                def key(x):
                    under = att[x] < quota(x)
                    return (not under, (0 if x in P1 else 1) if under else 0, att[x] / (2 if x in P1 else 1))
                l = min(cand, key=key); att[l] += 1; c = queues[l].pop(0); return l, build_prompt(l, rnd, c, B)
        def worker(_):
            while time.time() < deadline:
                j = next_job()
                if not j: return
                l, pr = j; V, real = Vs[l]
                try: text, ct, sec, fin = call(url, model, pr["system"], pr["user"], pr["max_new"])
                except Exception: time.sleep(5); continue
                m = metrics(text, V, real); ok = keep(m, pr["target_words"], thr.get(l, 1.6 if B[l]["tier"] == "low" else 2.0), l)       # low-resource languages: stricter invented-word limit
                rec = {"id": hashlib.sha1((l + text).encode()).hexdigest()[:12], "lang": l, "text": text.strip(), "topic": pr["topic"], "aspect": pr["aspect"], "place": pr["place"], "doc_type": pr["doc_type"], "card_id": pr["card_id"],
                       "model": model, "tier": B[l]["tier"], "seconds": round(sec, 1), "completion_tokens": ct, "finish": fin, **m, "keep": ok}
                with lk:
                    kept[l] += ok
                    with open(HERE / "out_cpu" / f"{l}.all.jsonl", "a", encoding="utf-8") as fa: fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if ok:
                        with open(HERE / "out_cpu" / f"{l}.jsonl", "a", encoding="utf-8") as fo: fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with ThreadPoolExecutor(conc) as ex: list(ex.map(worker, range(conc)))


if __name__ == "__main__":
    main()
