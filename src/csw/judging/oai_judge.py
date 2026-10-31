"""gpt-4o (or any OpenAI model) as a LABELLER through the OpenAI Batch API (50% price, 24 h window). The labels train the quality estimator later.
python -m csw.judging.oai_judge build  [--sets gold old gen] [--limit N] [--per-lang K] [--model gpt-4o]   -> oai/requests.jsonl  (+ cost estimate, nothing is sent)
python -m csw.judging.oai_judge submit                                                                      -> uploads requests.jsonl, creates the batch, writes oai/batch.json
python -m csw.judging.oai_judge poll                                                                        -> status; when done writes oai/labels.jsonl
python -m csw.judging.oai_judge report                                                                      -> agreement of the labels with what we know (gold = good, tagged-bad old rows)
Needs OPENAI_API_KEY (environment or the repo .env). Every label also says whether the model can actually READ the language (can_read) - rows it cannot read are not used as labels."""
from csw.paths import DATA_ROOT, env_value
import json, os, sys, time, hashlib, random, collections, statistics as st
from pathlib import Path
HERE = DATA_ROOT
OUT = HERE / "oai"; OUT.mkdir(exist_ok=True)
PRICE = {"gpt-4o": (1.25, 5.0), "gpt-4o-mini": (0.075, 0.30), "gpt-5.4": (1.25, 7.5), "gpt-5-mini": (0.125, 1.0)}   # batch $ / 1M tokens (in, out)

SYSTEM = """You are an expert reviewer of African-language text for a training dataset. Judge ONE sample. Be strict and honest: if you cannot read the language well enough to judge a field, say so - never guess and never reward fluent-looking text you cannot verify.

The sample has four fields:
- eng: formal, neutral English (no emotive words, idioms, metaphors, slang or contractions; same facts as the others).
- m_text: the official translation of eng into the target language (faithful, grammatical, real target language, standard orthography).
- hm: a natural code-switched variation of m_text: the target language is the base and must lead; only the PINNED switch language supplies words/short phrases (about 15-40% of the words); base-language grammar kept; full diacritics; no third language.
- ood: complete informal native parlance of the same situation: idiom, proverbs, imagery, ideophones; dominantly native, richer than hm, same facts.

Score each of eng, m_text, hm, ood from 1 to 10: 10 = publishable by a native expert unchanged; 8 = good, minor slips; 6 = understandable but clear errors; 4 = mostly wrong or partly another language; 1-2 = wrong language, copied, or meaningless. Check meaning against eng (wrong numbers, names, negation or facts are serious errors), invented or doubtful words, missing diacritics, wrong language, over- or under-mixing in hm, and flat, literal ood.

Return ONE JSON object:
{"can_read": true|false, "language_level": "fluent|good|limited|poor", "scores": {"eng": 0, "m_text": 0, "hm": 0, "ood": 0}, "verdict": "keep|fix|drop", "problems": ["short English notes: field + what is wrong"], "confidence": 0.0}
can_read=false (or language_level poor) means your scores for the target-language fields are not reliable."""


def key():
    k = env_value("OPENAI_API_KEY")
    if not k: raise SystemExit("OPENAI_API_KEY missing - add it to .env")
    return k


def card(r):
    from csw.datagen.profiles import P
    p = P[r["lang"]]
    sc = r.get("scenario") or {}
    return (f"TARGET LANGUAGE: {p.name} [{p.code}] | PINNED SWITCH LANGUAGE: {sc.get('switch') or r.get('switch')} | DIALECT: {sc.get('dialect') or r.get('dialect') or 'standard'} | GENRE: {sc.get('genre') or r.get('genre')}\n"
            f"ORTHOGRAPHY: {p.orthography}\n" + json.dumps({k: r[k] for k in ("eng", "m_text", "hm", "ood")}, ensure_ascii=False, indent=1))


def rows(sets, per_lang):
    from csw.datagen import cs_prompts as pr
    out = []
    if "gold" in sets:                                     # known-good native-corrected rows
        for l, rs in pr.load_aug_gold().items():
            for r in rs[:per_lang or 999]: out.append(dict(r, _set="gold", _id=f"gold-{l}-{hashlib.sha1((r['eng']+r['m_text']+r['hm']+r['ood']).encode()).hexdigest()[:10]}", lang=l))
    if "old" in sets:                                      # the 1,188 old generated rows, with their rule scores / tags
        by = collections.defaultdict(list)
        for line in (HERE / "old_generated" / "old_generated_v3.jsonl").read_text(encoding="utf-8").split("\n"):
            if line.strip(): r = json.loads(line); by[r["lang"]].append(r)
        for l, rs in by.items():
            for r in rs[:per_lang or 999]: out.append(dict(r, _set="old", _id=f"old-{r['id']}"))
    if "gen" in sets:                                      # new generated rows (out_main raw files)
        for f in sorted((HERE / "out_main").glob("*.raw.jsonl")):
            n = 0
            for line in f.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    r = json.loads(line)
                    if all(r.get(k) for k in ("eng", "m_text", "hm", "ood")) and n < (per_lang or 999): out.append(dict(r, _set="gen", _id=f"gen-{r['id']}")); n += 1
    seen, uniq = set(), []
    for r in out:                                          # custom_id must be unique inside a batch; exact duplicate rows are judged once
        if r["_id"] not in seen: seen.add(r["_id"]); uniq.append(r)
    return uniq


def build(a):
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    model = opt("--model", "gpt-4o"); sets = []
    if "--sets" in a:
        i = a.index("--sets") + 1
        while i < len(a) and not a[i].startswith("--"): sets.append(a[i]); i += 1
    sets = sets or ["gold", "old", "gen"]
    rs = rows(sets, int(opt("--per-lang", 0)))
    if opt("--limit"): random.Random(0).shuffle(rs); rs = rs[: int(opt("--limit"))]
    reqs = [{"custom_id": r["_id"], "method": "POST", "url": "/v1/chat/completions",
             "body": {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": card(r)}], "temperature": 0, "max_tokens": 500, "response_format": {"type": "json_object"}}} for r in rs]
    (OUT / "requests.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in reqs) + "\n", encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps({r["_id"]: {"set": r["_set"], "lang": r["lang"], "rule_score": ((r.get("scores") or {}).get("rule_score") if isinstance(r.get("scores"), dict) else None), "tags": r.get("quality_tags")} for r in rs}, ensure_ascii=False))
    tin = sum(len(x["body"]["messages"][0]["content"]) + len(x["body"]["messages"][1]["content"]) for x in reqs) / 2.6     # African-language text tokenizes badly: ~2.6 chars/token
    pi, po = PRICE.get(model, (1.25, 5.0)); tout = len(reqs) * 220
    print(f"{len(reqs)} requests ({collections.Counter(r['_set'] for r in rs)}) | ~{tin/1e6:.2f}M input + ~{tout/1e6:.2f}M output tokens | batch cost for {model}: ~${tin/1e6*pi + tout/1e6*po:.2f}")


def req(method, path, **kw):
    import requests
    r = requests.request(method, "https://api.openai.com/v1" + path, headers={"Authorization": f"Bearer {key()}", **kw.pop("headers", {})}, timeout=300, **kw)
    if r.status_code >= 300: raise SystemExit(f"{method} {path}: {r.status_code} {r.text[:400]}")
    return r


def submit():
    f = req("POST", "/files", data={"purpose": "batch"}, files={"file": ("requests.jsonl", (OUT / "requests.jsonl").read_bytes())}).json()
    b = req("POST", "/batches", json={"input_file_id": f["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"}).json()
    (OUT / "batch.json").write_text(json.dumps(b)); print("batch", b["id"], b["status"])


def poll():
    b = req("GET", "/batches/" + json.loads((OUT / "batch.json").read_text())["id"]).json()
    print(b["status"], b.get("request_counts"))
    if b["status"] == "completed" and b.get("output_file_id"):
        (OUT / "labels.raw.jsonl").write_bytes(req("GET", "/files/" + b["output_file_id"] + "/content").content)
        out = []
        for line in (OUT / "labels.raw.jsonl").read_text(encoding="utf-8").split("\n"):
            if line.strip():
                o = json.loads(line)
                try: j = json.loads(o["response"]["body"]["choices"][0]["message"]["content"])
                except Exception: j = None
                out.append({"id": o["custom_id"], "label": j, "usage": (o.get("response") or {}).get("body", {}).get("usage")})
        (OUT / "labels.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8"); print("wrote", len(out), "labels")


def report():
    meta = json.loads((OUT / "meta.json").read_text()); L = [json.loads(l) for l in (OUT / "labels.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    ok = [x for x in L if x["label"] and isinstance(x["label"].get("scores"), dict)]
    print(len(L), "labels,", len(ok), "parsed")
    by = collections.defaultdict(list)
    for x in ok: by[(meta[x["id"]]["lang"], meta[x["id"]]["set"])].append(x["label"])
    mean = lambda v: round(st.mean(v), 1) if v else "-"
    langs = sorted({k[0] for k in by})
    print(f"{'lang':5}{'gold m/h/o':>16}{'old m/h/o':>16}{'gen m/h/o':>16}  can_read(gold)  levels(gold)")
    for l in langs:
        row = [f"{l:5}"]
        for s in ("gold", "old", "gen"):
            v = by.get((l, s), [])
            row.append(f"{mean([x['scores'].get('m_text') for x in v if isinstance(x['scores'].get('m_text'), (int, float))])}/{mean([x['scores'].get('hm') for x in v if isinstance(x['scores'].get('hm'), (int, float))])}/{mean([x['scores'].get('ood') for x in v if isinstance(x['scores'].get('ood'), (int, float))])}".rjust(16))
        g = by.get((l, "gold"), []); row.append(f"  {sum(bool(x.get('can_read')) for x in g)}/{len(g)}   {dict(collections.Counter(x.get('language_level') for x in g))}")
        print("".join(row))


if __name__ == "__main__":
    {"build": lambda: build(sys.argv[2:]), "submit": submit, "poll": poll, "report": report}[sys.argv[1]]()
