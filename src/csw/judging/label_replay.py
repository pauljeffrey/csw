"""Flash-label the human replay pairs (English -> African), M1..M6 only, very short prompt, 8 same-language pairs per request, Batch API (50% price).
python -m csw.judging.label_replay build   --pairs work/replay2.jsonl --out work/replay_flash [--per 8] [--shard 700]
python -m csw.judging.label_replay count   --out work/replay_flash                  # token count of a sample -> cost estimate (no generation)
python -m csw.judging.label_replay submit  --out work/replay_flash [--max-usd 7]
python -m csw.judging.label_replay collect --out work/replay_flash                  # -> work/replay_flash/labels.jsonl {id, lang, eng, m, g:[M1..M6]}"""
from csw.paths import DATA_ROOT
import argparse, collections, hashlib, json, random, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.sft import fmt
from csw.judging import gemini_batch as GB

SYSTEM = ("Grade each English->{lang} translation 1-5 (null if you cannot judge): M1 faithfulness (numbers, names, negation, omission, addition); M2 fluency/grammar; M3 orthography/diacritics; "
          "M4 language identity (no other language); M5 neutral formal register; M6 terminology/names. Reply only JSON {{\"g\":[[M1..M6],...]}}, one list per item, same order.")
STRICT_M = (" Grade as a strict native reviewer: 5 = no flaw a native would notice; 4 = exactly one minor flaw; 3 = several minor flaws or one clear flaw; 2 = a major flaw; 1 = unusable. "
            "Do not default to 4 or 5: about half the grades in a typical batch are 3 or lower. Untranslated, garbled, looping, truncated or wrong-language output is 1-2 on the dimensions it breaks.")
RULES_M = (" Items may carry r = computed checks that failed, trust them: num = numbers/digits of en missing or changed in t (M1 <= 2); short/long = t much shorter/longer than en (check omitted or added content, M1 <= 3); "
           "loop = t repeats itself (M2 <= 2).")
IN_USD, OUT_USD = 0.375, 1.875                           # Gemini 3.8 Flash, Batch API, per 1M tokens (reports/judge_flash_runs_2026-10-25.md)


def pid(x): return x["uid"] if x.get("uid") else "rp-" + hashlib.sha1((x["lang"] + "|" + " ".join(x["eng"].split())).encode()).hexdigest()[:12]


def rules(en, t):
    import re, unicodedata
    r = {}; ne = re.findall(r"\d+(?:[.,]\d+)*", en); nt = set(re.findall(r"\d+(?:[.,]\d+)*", t))
    miss = [x for x in ne if x not in nt]
    if miss: r["num"] = miss[:4]
    ratio = len(t) / max(1, len(en))
    if ratio < 0.55: r["short"] = round(ratio, 2)
    elif ratio > 2.2: r["long"] = round(ratio, 2)
    w = t.split()
    if len(w) >= 12:
        g = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
        if len(set(g)) / len(g) < 0.6: r["loop"] = True
    return r


def build(a):
    base_sys = SYSTEM + (STRICT_M if getattr(a, "strict", False) else "") + (RULES_M if getattr(a, "rules", False) else "")
    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8") if l.strip()]; by = collections.defaultdict(list)
    for x in pairs: by[x["lang"]].append(x)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); reqs = []; meta = {}
    for lang, xs in sorted(by.items()):
        random.Random(lang).shuffle(xs)
        for k in range(0, len(xs), a.per):
            grp = xs[k:k + a.per]; key = f"rl-{lang}-{k // a.per:05d}"; meta[key] = [pid(x) for x in grp]
            user = json.dumps([dict({"en": x["eng"][:600], "t": x["m"][:900]}, **({"r": rr} if getattr(a, "rules", False) and (rr := rules(x["eng"], x["m"])) else {})) for x in grp], ensure_ascii=False)
            reqs.append({"key": key, "request": {"contents": [{"role": "user", "parts": [{"text": user}]}], "system_instruction": {"parts": [{"text": base_sys.format(lang=fmt.NAMES.get(lang, lang))}]},
                         "generation_config": {"temperature": 0, "max_output_tokens": 800, "response_mime_type": "application/json", "thinking_config": {"thinking_level": "low"}}}})
    for s in range(0, len(reqs), a.shard):
        d = out / f"stage_s{s // a.shard:02d}"; d.mkdir(exist_ok=True)
        with open(d / "gemini.jsonl", "w", encoding="utf-8") as f:
            for r in reqs[s:s + a.shard]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps(meta)); (out / "pairs.jsonl").write_text("\n".join(json.dumps(dict(x, id=pid(x)), ensure_ascii=False) for x in pairs))
    print(len(pairs), "pairs ->", len(reqs), "requests in", -(-len(reqs) // a.shard), "shards")


def count(a):
    c = GB.client(); allr = [json.loads(l) for d in sorted(Path(a.out).glob("stage_s*")) for l in open(d / "gemini.jsonl", encoding="utf-8")]; rnd = random.Random(1); sm = rnd.sample(allr, min(40, len(allr))); tot_in = 0
    for r in sm:
        rq = r["request"]; txt = rq["system_instruction"]["parts"][0]["text"] + "\n" + rq["contents"][0]["parts"][0]["text"]; n = c.models.count_tokens(model="models/gemini-3.8-flash", contents=txt).total_tokens; tot_in += n
    avg_in = tot_in / len(sm); n = len(allr); out_tok = 330                                  # 8 grade lists (~100 tokens) + low thinking, measured ~300 on earlier Flash runs
    est = n * (avg_in * IN_USD + out_tok * OUT_USD) / 1e6; worst = n * (avg_in * IN_USD + 800 * OUT_USD) / 1e6
    print(f"requests {n}; mean input {avg_in:.0f} tokens; estimate ${est:.2f} (output ~{out_tok} tok/request); worst case at the 800-token cap ${worst:.2f}")
    (Path(a.out) / "estimate.json").write_text(json.dumps({"requests": n, "avg_in": avg_in, "estimate_usd": est, "worst_usd": worst}))


def submit(a):
    est = json.loads((Path(a.out) / "estimate.json").read_text())
    if est["worst_usd"] > a.max_usd: raise SystemExit(f"worst case ${est['worst_usd']:.2f} exceeds the cap ${a.max_usd}")
    for d in sorted(Path(a.out).glob("stage_s*")):
        if (d / "gemini_batch.json").exists(): continue
        GB.submit(argparse.Namespace(stage=str(d), model="gemini-3.8-flash", name=f"replay-{d.name}"))


def collect(a):
    out = Path(a.out); meta = json.loads((out / "meta.json").read_text()); pairs = {p["id"]: p for p in map(json.loads, (out / "pairs.jsonl").read_text().split("\n"))}; n = bad = 0; usage = [0, 0]
    with open(out / "labels.jsonl", "w", encoding="utf-8") as fo:
        for d in sorted(out.glob("stage_s*")):
            GB.poll(argparse.Namespace(stage=str(d)))
            gen = out / "results" / "stage_gemini-3.8-flash.jsonl"                 # gemini_batch.poll writes every shard to this one name: keep a copy per shard
            if gen.exists(): (out / "results" / f"{d.name}.jsonl").write_text(gen.read_text(encoding="utf-8"), encoding="utf-8")
        for f in sorted((out / "results").glob("stage_s*.jsonl")) if (out / "results").exists() else []:
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); ids = meta.get(r["id"]); g = (r.get("parsed") or {}).get("g") if isinstance(r.get("parsed"), dict) else None
                u = r.get("usage") or {}; usage[0] += u.get("promptTokenCount", 0); usage[1] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
                if not ids or not isinstance(g, list) or len(g) != len(ids): bad += 1; continue
                for i, v in zip(ids, g):
                    if isinstance(v, list) and len(v) == 6 and all(x is None or (isinstance(x, (int, float)) and 1 <= x <= 5) for x in v):
                        p = pairs[i]; fo.write(json.dumps({"id": i, "lang": p["lang"], "eng": p["eng"], "m": p["m"], "origin": p.get("origin"), "g": v}, ensure_ascii=False) + "\n"); n += 1
    print(f"labelled pairs {n}; unusable requests {bad}; tokens in {usage[0]:,} out {usage[1]:,}; cost ${(usage[0] * IN_USD + usage[1] * OUT_USD) / 1e6:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("build", "count", "submit", "collect"):
        p = sp.add_parser(n); p.add_argument("--out", required=True); p.add_argument("--pairs", default="work/replay2.jsonl"); p.add_argument("--per", type=int, default=8); p.add_argument("--shard", type=int, default=700); p.add_argument("--rules", action="store_true"); p.add_argument("--max-usd", type=float, default=7.0); p.add_argument("--strict", action="store_true")
    a = ap.parse_args(); {"build": build, "count": count, "submit": submit, "collect": collect}[a.cmd](a)
