"""Does the gpt-4o judge CATCH real errors?  Builds corrupted copies of gold rows that gpt-4o already rated well, sends them through the same judge, and counts detections.
python -m csw.judging.oai_corrupt build [--n 200]   -> oai_corrupt/requests.jsonl (+ meta with the corruption type)   [nothing is sent]
python -m csw.judging.oai_corrupt submit | poll     -> same Batch API flow as oai_judge.py, files under oai_corrupt/
python -m csw.judging.oai_corrupt report            -> detection rate per corruption type vs the clean controls
Corruptions (the target-language text is never edited, so no language knowledge is needed):
  clean        - unchanged row (false-alarm rate)            number    - one number word in eng changed (eng no longer matches the translations)
  swap         - m_text/hm/ood replaced by another row's     negation  - one negation in eng flipped
  truncate     - m_text/hm/ood cut to ~45% of the words      wronglang - m_text/hm/ood taken from an unrelated language's row"""
from csw.paths import DATA_ROOT
import json, re, sys, random, hashlib, collections, statistics as st
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import oai_judge as oj
OUT = HERE / "oai_corrupt"; OUT.mkdir(exist_ok=True); oj.OUT = OUT
NUM = {"one": "three", "two": "five", "three": "seven", "four": "nine", "five": "eight", "six": "two", "seven": "four", "eight": "three", "nine": "six", "ten": "twenty", "twenty": "ten", "fifty": "five",
       "hundred": "thousand", "thousand": "hundred", "million": "thousand", "first": "third", "second": "fifth", "half": "double", "weekly": "daily", "daily": "monthly", "monthly": "weekly"}
NEG = [(r"\bdo not\b", "do"), (r"\bdoes not\b", "does"), (r"\bdid not\b", "did"), (r"\bcannot\b", "can"), (r"\bmust not\b", "must"), (r"\bshould not\b", "should"), (r"\bwill not\b", "will"),
       (r"\bnever\b", "always"), (r"\bno one\b", "everyone"), (r"\bis not\b", "is"), (r"\bare not\b", "are"), (r"\bwas not\b", "was"), (r"\bhave not\b", "have"), (r"\bhas not\b", "has")]


def change_number(eng, rnd):
    ms = [m for m in re.finditer(r"\b(" + "|".join(NUM) + r")\b", eng, re.I)]
    if not ms:
        ds = list(re.finditer(r"\b\d+\b", eng))
        if not ds: return None
        m = rnd.choice(ds); return eng[:m.start()] + str(int(m.group()) + rnd.choice([2, 3, 5])) + eng[m.end():]
    m = rnd.choice(ms); w = NUM[m.group().lower()]
    return eng[:m.start()] + (w.capitalize() if m.group()[0].isupper() else w) + eng[m.end():]


def flip_negation(eng, rnd):
    hits = [(re.search(p, eng, re.I), r) for p, r in NEG]; hits = [(m, r) for m, r in hits if m]
    if not hits: return None
    m, r = rnd.choice(hits); return eng[:m.start()] + (r.capitalize() if m.group()[0].isupper() else r) + eng[m.end():]


def truncate(t):
    w = t.split(); return " ".join(w[: max(4, int(len(w) * 0.45))])


def build(n=200):
    from csw.datagen import cs_prompts as pr
    rnd = random.Random(7)
    base = json.loads((HERE / "oai" / "meta.json").read_text()); labs = {}
    for line in (HERE / "oai" / "labels.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip(): o = json.loads(line); labs[o["id"]] = o["label"]
    gold = [dict(r, lang=l) for l, rs in pr.load_aug_gold().items() for r in rs]
    good = []
    for r in gold:                                          # only rows gpt-4o already rated well AND could read: any drop is then caused by the corruption
        gid = f"gold-{r['lang']}-{hashlib.sha1((r['eng'] + r['m_text'] + r['hm'] + r['ood']).encode()).hexdigest()[:10]}"; lab = labs.get(gid)
        if lab and lab.get("can_read") and isinstance(lab.get("scores"), dict) and all(isinstance(lab["scores"].get(k), (int, float)) and lab["scores"][k] >= 8 for k in ("m_text", "hm", "ood")): good.append((r, lab))
    print(len(good), "gold rows rated >= 8 on all fields and readable by gpt-4o, in", len({r['lang'] for r, _ in good}), "languages")
    rnd.shuffle(good); by_lang = collections.defaultdict(list)
    for r, lab in good: by_lang[r["lang"]].append(r)
    plan = [("clean", 40), ("swap", 40), ("truncate", 30), ("wronglang", 30), ("number", 30), ("negation", 30)]
    reqs, meta, used = [], {}, set()
    pool = [r for r, _ in good]
    for kind, k in plan:
        made = 0
        for r in pool:
            if made >= k: break
            key = (kind, r["eng"] + r["lang"])
            if key in used: continue
            c = dict(r, scenario={"switch": r["switch"], "dialect": r.get("dialect"), "genre": r["genre"]})
            if kind == "swap":
                o = [x for x in by_lang[r["lang"]] if x["eng"] != r["eng"]]
                if not o: continue
                x = rnd.choice(o); c.update(m_text=x["m_text"], hm=x["hm"], ood=x["ood"])
            elif kind == "truncate":
                c.update(m_text=truncate(r["m_text"]), hm=truncate(r["hm"]), ood=truncate(r["ood"]))
            elif kind == "wronglang":
                o = [x for x in pool if x["lang"] not in (r["lang"], "gpe", "wes", "pcm") and x["lang"] in ("swh", "hau", "zul", "som", "amh", "afr", "yor", "kin", "sna", "mlg")]
                if not o: continue
                x = rnd.choice(o); c.update(m_text=x["m_text"], hm=x["hm"], ood=x["ood"])
            elif kind == "number":
                e = change_number(r["eng"], rnd)
                if not e: continue
                c["eng"] = e
            elif kind == "negation":
                e = flip_negation(r["eng"], rnd)
                if not e: continue
                c["eng"] = e
            used.add(key); made += 1
            cid = f"cor-{kind}-{hashlib.sha1((kind + c['eng'] + c['m_text'] + c['lang']).encode()).hexdigest()[:10]}"
            reqs.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": oj.SYSTEM}, {"role": "user", "content": oj.card(c)}], "temperature": 0, "max_tokens": 500, "response_format": {"type": "json_object"}}})
            meta[cid] = {"type": kind, "lang": r["lang"]}
        print(kind, "built", made, "of", k)
    (OUT / "requests.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in reqs) + "\n", encoding="utf-8"); (OUT / "meta.json").write_text(json.dumps(meta))
    print(len(reqs), "requests ->", OUT / "requests.jsonl")


def report():
    meta = json.loads((OUT / "meta.json").read_text()); L = [json.loads(l) for l in (OUT / "labels.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    by = collections.defaultdict(list)
    for x in L:
        lab = x["label"]
        if lab and isinstance(lab.get("scores"), dict): by[meta[x["id"]]["type"]].append(lab)
    print(f"{'corruption':11}{'n':>4} {'caught(verdict fix/drop)':>26} {'caught(any score<=6)':>22} {'mean m/h/o':>16}")
    for t in ("clean", "swap", "truncate", "wronglang", "number", "negation"):
        v = by.get(t, [])
        if not v: continue
        sc = lambda k: st.mean([x["scores"].get(k) for x in v if isinstance(x["scores"].get(k), (int, float))])
        a = sum(x.get("verdict") in ("fix", "drop") for x in v); b = sum(min(x["scores"].get(k, 10) for k in ("m_text", "hm", "ood")) <= 6 for x in v)
        print(f"{t:11}{len(v):>4} {a:>10} = {100*a/len(v):>4.0f}% {b:>16} = {100*b/len(v):>4.0f}% {sc('m_text'):>7.1f}/{sc('hm'):.1f}/{sc('ood'):.1f}")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build": build()
    elif c == "submit": oj.submit()
    elif c == "poll": oj.poll()
    elif c == "report": report()
