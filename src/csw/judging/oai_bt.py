"""Back-translate-then-compare: gpt-4o translates each m_text of the corrupted test set back into English (it never sees eng), then number and negation are compared with eng by RULE.
python -m csw.judging.oai_bt build | submit | poll | report      (files in oai_bt/; reuses the 200 rows of oai_corrupt/)"""
from csw.paths import DATA_ROOT
import json, re, sys, collections
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import oai_judge as oj
OUT = HERE / "oai_bt"; OUT.mkdir(exist_ok=True); oj.OUT = OUT
SYS = """You are a careful translator. Translate the given text from {lang} into English, sentence by sentence, as literally and completely as you can. Do NOT correct, improve, summarise or complete it: if the text says something odd or wrong, translate exactly that. Keep every number, name and negation exactly as the text says it. If a passage is unreadable to you, write [unreadable] for it. Return JSON: {{"english": "..."}}"""
UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
         "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "half": 0.5, "double": 2, "twice": 2}
MULT = {"hundred": 100, "thousand": 1000, "million": 10 ** 6}
NEGW = re.compile(r"\b(not|no|never|cannot|nobody|nothing|none|neither|nor|without|n't)\b|n't\b", re.I)


def numbers(t: str) -> list:
    out, cur, have = [], 0, False
    toks = re.findall(r"\d[\d,]*\.?\d*|[A-Za-z']+", t.replace("-", " "))
    def flush():
        nonlocal cur, have
        if have: out.append(cur)
        cur, have = 0, False
    for w in toks:
        lw = w.lower()
        if re.fullmatch(r"\d[\d,]*\.?\d*", w):
            flush(); out.append(float(w.replace(",", ""))); continue
        if lw in UNITS and UNITS[lw] >= 1: cur += UNITS[lw]; have = True
        elif lw in MULT and have: cur = (cur or 1) * MULT[lw]
        elif lw in MULT: cur, have = MULT[lw], True
        else: flush()
    flush(); return [x for x in out if x]


def negs(t: str) -> int:
    return len(NEGW.findall(t))


def build():
    meta = json.loads((HERE / "oai_corrupt" / "meta.json").read_text()); reqs = []; rows = {}
    for line in (HERE / "oai_corrupt" / "requests.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip(): continue
        r = json.loads(line); u = r["body"]["messages"][1]["content"]; d = json.loads(u[u.index("{\n"):])
        lang = meta[r["custom_id"]]["lang"]; rows[r["custom_id"]] = {"type": meta[r["custom_id"]]["type"], "lang": lang, "eng": d["eng"], "m_text": d["m_text"]}
        from csw.datagen.profiles import P
        reqs.append({"custom_id": r["custom_id"], "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "temperature": 0, "max_tokens": 1500, "response_format": {"type": "json_object"},
                     "messages": [{"role": "system", "content": SYS.format(lang=P[lang].name)}, {"role": "user", "content": d["m_text"]}]}})
    (OUT / "requests.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in reqs) + "\n", encoding="utf-8"); (OUT / "rows.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(len(reqs), "back-translation requests")


def report():
    rows = json.loads((OUT / "rows.json").read_text()); L = {json.loads(l)["id"]: json.loads(l) for l in (OUT / "labels.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()}
    res = collections.defaultdict(lambda: collections.Counter())
    for cid, r in rows.items():
        bt = ((L.get(cid) or {}).get("label") or {}).get("english") or ""
        ne, nb = numbers(r["eng"]), numbers(bt); ge, gb = negs(r["eng"]), negs(bt)
        num_flag = bool(ne) and sorted(ne) != sorted(x for x in nb if x in ne or True) and not all(x in nb for x in ne)       # an eng number is missing from the back-translation
        neg_flag = (ge > 0) != (gb > 0) or abs(ge - gb) >= 2                                                                    # negation present in one side only (or clearly different)
        unread = "[unreadable]" in bt
        c = res[r["type"]]; c["n"] += 1; c["number"] += num_flag; c["negation"] += neg_flag; c["either"] += (num_flag or neg_flag); c["unreadable"] += unread
        c["has_numbers"] += bool(ne)
    print(f"{'corruption':11}{'n':>4} {'number flag':>12} {'negation flag':>14} {'either':>8} {'eng has numbers':>16} {'[unreadable]':>13}")
    for t in ("clean", "number", "negation", "swap", "truncate", "wronglang"):
        c = res[t]
        if c["n"]: print(f"{t:11}{c['n']:>4} {c['number']:>7} ({100*c['number']/c['n']:>3.0f}%) {c['negation']:>8} ({100*c['negation']/c['n']:>3.0f}%) {c['either']:>4} ({100*c['either']/c['n']:>3.0f}%) {c['has_numbers']:>10} {c['unreadable']:>13}")
    ex = [(cid, r) for cid, r in rows.items() if r["type"] in ("number", "negation")][:0]


if __name__ == "__main__":
    c = sys.argv[1]
    {"build": build, "submit": oj.submit, "poll": oj.poll, "report": report}[c]()
