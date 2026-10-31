"""Code-switching corpus pipeline: anchors -> seeds -> plan -> generate -> judge -> finalize.

  python -m csw.datagen.cs_pipeline anchors --langs yor hau
  python -m csw.datagen.cs_pipeline seeds   --langs yor hau
  python -m csw.datagen.cs_pipeline generate --langs yor hau --n 20 --model google/gemma-4-31b-it
  python -m csw.datagen.cs_pipeline judge   --langs yor hau --judge-model google/gemini-3.1-pro-preview
  python -m csw.datagen.cs_pipeline report  --langs yor hau

Everything is resumable: each stage appends JSONL keyed by id and skips ids already present.
"""
from __future__ import annotations

from csw.paths import DATA_ROOT
import argparse
import collections
import hashlib
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = DATA_ROOT

from csw.datagen import gates  # noqa: E402
from csw.datagen import cs_prompts as prompts  # noqa: E402
from csw.datagen import seeds as seedmod  # noqa: E402
from csw.datagen.profiles import COMPAT, GENRES, LANGS, LENGTHS, P, TRACKS, TARGETS, BRIEF, SENT_RANGE, SKIP_NATIVE_SEEDS, REUSE_FROM, REUSE_SHARE, REUSE_DEFAULT_SOURCES  # noqa: E402
from csw.datagen.providers import Request, get_provider  # noqa: E402

import os
OUT = HERE / os.environ.get("CS_OUT", "out")
ANCH = HERE / "anchors"
OUT.mkdir(exist_ok=True)
ANCH.mkdir(exist_ok=True)
_lock = threading.Lock()

PERSONAS = ["young woman, 20s, university student", "young man, 20s, hustler/tech worker", "middle-aged mother, trader", "middle-aged father, civil servant",
            "elder grandmother, storyteller", "elder grandfather, community figure", "teenager, secondary school", "nurse or doctor, 30s", "farmer, 40s",
            "commercial driver or conductor, 30s", "pastor / imam / traditional priest", "diaspora professional visiting home"]
REGISTERS = ["casual/intimate", "casual/street", "respectful (elder or stranger)", "semi-formal (office, school)", "humorous/teasing", "heated/argumentative", "tender/consoling"]
EMOTIONS = ["amused", "worried", "angry", "proud", "nostalgic", "excited", "disappointed", "grateful", "sarcastic", "calm and instructive", "grieving", "curious"]
AUDIENCES = ["a close friend", "a younger sibling or cousin", "a parent or elder relative", "a spouse or partner", "a colleague or boss", "a customer or trader", "a neighbour", "a child or grandchild", "a group chat or family WhatsApp group", "a radio or phone-in audience", "a stranger or official", "a teacher or student"]
MEDIA = ["WhatsApp text", "voice note", "face-to-face talk", "phone call", "radio/phone-in", "social-media comment"]
DOMAINS = ["money and informal finance", "health and body", "family and marriage", "school and learning", "work and hustle", "farming and food", "religion and faith",
           "politics and community", "technology and phones", "travel and transport", "sports and entertainment", "law, disputes and custom", "weather, land and animals", "identity, migration and generation gap"]


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()] if path.exists() else []


def append(path: Path, row: dict):
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wchoice(rnd, table, idx=0):
    ks = list(table)
    return rnd.choices(ks, weights=[v[idx] if isinstance(v, tuple) else v for v in table.values()])[0]


GOLD: list = []
GEN_TOP_P = float(os.environ.get("CS_TOP_P", 0.95))
GEN_TEMP = float(os.environ.get("CS_TEMP", 0.6))     # 0.6: accuracy over variety (variety comes from the stratified plan)
PROMPT_FP = ""            # fingerprint of the prompts a run used (stored on every row)
PROVIDER = "openrouter"


def _local_provider(model: str, conc: int):
    """OpenAI-compatible server on this machine (vLLM): CS_LOCAL_URL, default http://127.0.0.1:8000/v1."""
    from csw.datagen.providers.base import Provider

    class LocalProvider(Provider):
        name = "local"
        base_url = os.environ.get("CS_LOCAL_URL", "http://127.0.0.1:8000/v1")
        env_key = "CS_LOCAL_KEY"
        default_pricing = (0.0, 0.0)
    os.environ.setdefault("CS_LOCAL_KEY", "local")
    return LocalProvider(model, max_concurrency=conc, timeout=int(os.environ.get('CS_TIMEOUT', 600)), max_retries=3)     # a request vLLM dropped must not hold a worker for 30 min


def prov(model: str, conc: int):
    pv = _local_provider(model, conc) if PROVIDER == "local" else get_provider(PROVIDER, model, max_concurrency=conc, timeout=600)
    if PROVIDER == "local":                                   # repetition loops were seen in weak languages; keep sampling tight
        orig_l = pv._payload
        extra = {"repetition_penalty": float(os.environ.get("CS_REP_PEN", 1.05)), "top_k": 40}
        if os.environ.get("CS_THINK"):                        # optional thinking mode (untested on Gemma-4: verify the server's chat template supports it)
            extra["chat_template_kwargs"] = {"enable_thinking": True}
        pv._payload = lambda req: (lambda p: {**p, "extra_body": {**p.get("extra_body", {}), **extra}})(orig_l(req))
    eff = os.environ.get("CS_REASONING")           # e.g. "none": turns thinking off on Doubleword (28 vs 704 tokens measured)
    if eff and PROVIDER == "doubleword":
        orig = pv._payload
        pv._payload = lambda req: {**orig(req), "reasoning_effort": eff}
    elif eff == "none" and PROVIDER == "openrouter":      # OpenRouter's unified switch (measured: 381 -> 28 tokens)
        orig = pv._payload
        pv._payload = lambda req: (lambda p: {**p, "extra_body": {**p.get("extra_body", {}), "reasoning": {"enabled": False}}})(orig(req))
    return pv


def load_exemplars(lang: str) -> list[dict]:
    return jl(HERE / "exemplars" / f"{lang}.jsonl")


def pick_exemplars(lang: str, c: dict, k: int = 3) -> list[dict]:
    """Hand-written gold examples first (same language), then model-built ones (same language); a language with
    neither borrows gold examples from OTHER languages, tagged so the model copies depth and structure, not words."""
    rnd = random.Random(c["id"])
    gold = jl(HERE / "exemplars" / "gold.jsonl")
    own = [dict(e, language=P[lang].name) for e in gold if e["lang"] == lang]
    built = [e for e in load_exemplars(lang)]
    rnd.shuffle(built)
    picked = (own + built)[:k]
    if not own and len(picked) < 2:
        other = [dict(e, language=P[e["lang"]].name) for e in gold if e["lang"] != lang]
        rnd.shuffle(other)
        picked += other[: 2 - len(picked)] if picked else other[:2]
    return picked


# ---------------- stage 1: anchors ----------------
def do_anchors(lang: str, model: str, per_cat: int, conc: int, rnd_no: int = 1):
    p, path = P[lang], ANCH / f"{lang}.jsonl"
    old = jl(path)
    have = {r["cat"] for r in old if r.get("round", 1) == rnd_no}
    seen_terms = {r["term"].lower() for r in old}
    pv = prov(model, conc)
    todo = [c for c in prompts.ANCHOR_CATS if c not in have]

    def one(cat):
        r = pv.complete(Request(f"anc-{lang}-{cat}", [{"role": "system", "content": prompts.ANCHOR_SYS},
                                                       {"role": "user", "content": prompts.anchor_prompt(p, cat, per_cat) + (" Avoid the obvious, famous items; prefer regional specifics, everyday objects, lesser-known practices and sayings." if rnd_no > 1 else "")}],
                                max_tokens=6000, temperature=0.7, response_format={"type": "json_object"}))
        d = prompts.parse_json(r.text) if r.ok else None
        items = [i for i in (d or {}).get("items", []) if i.get("term") and i.get("gloss")]
        items = [i for i in items if i["term"].lower() not in seen_terms]
        for i in items:
            seen_terms.add(i["term"].lower())
            append(path, {"lang": lang, "cat": cat, "round": rnd_no, **i})
        return cat, len(items), r.prompt_tokens, r.completion_tokens
    with ThreadPoolExecutor(conc) as ex:
        res = [f.result() for f in as_completed([ex.submit(one, c) for c in todo])]
    print(lang, "anchors:", sum(r[1] for r in res), "new;", "tokens", sum(r[2] for r in res), sum(r[3] for r in res))


# ---------------- stage 2: plan ----------------
MAX_SEED_REUSE = 3


def _quota(rnd, table, n):
    """Exact counts by largest remainder, shuffled: stratified, not iid, so a 6k plan hits its targets to +-1."""
    ks = list(table)
    ws = [table[k][0] if isinstance(table[k], tuple) else table[k] for k in ks]
    tot = sum(ws)
    raw = [n * w / tot for w in ws]
    cnt = [int(r) for r in raw]
    for i in sorted(range(len(ks)), key=lambda i: raw[i] - cnt[i], reverse=True)[: n - sum(cnt)]:
        cnt[i] += 1
    out = [k for k, c_ in zip(ks, cnt) for _ in range(c_)]
    rnd.shuffle(out)
    return out


def make_plan(lang: str, n: int, seed: int | None = None) -> list[dict]:
    """Stratified scenario plan: exact genre / length / domain / switch / dialect quotas; seeds are chosen
    least-used-first from the pool compatible with the row's genre; thin genres with no compatible seed use anchors."""
    seed = int(os.environ.get("CS_SEED", 0)) if seed is None else seed
    p, rnd = P[lang], random.Random(f"{lang}-{seed}")
    anchors = jl(ANCH / f"{lang}.jsonl")
    sd = seedmod.load(lang)
    nat = [] if lang in SKIP_NATIVE_SEEDS else [s for s in sd if s["track"] == "B"]
    eng_stories = [s for s in sd if s["track"] == "C"] + [s for s in seedmod.load("_en") if s["track"] == "C"] + seedmod.load("_en_simple")
    en_dom = collections.defaultdict(list)                               # English domain seeds (money / law / health / tech / ...)
    for s in seedmod.load("_en_domain"):
        en_dom[s["domain"]].append(s)
    eng_pool = jl(HERE / "eng_pool.jsonl") if (lang in REUSE_SHARE) else []   # Track D: English already generated for OTHER languages
    srcs = REUSE_FROM.get(lang, REUSE_DEFAULT_SOURCES)
    eng_pool = [e for e in eng_pool if e["lang"] in srcs and e["lang"] != lang]
    rC, rD = random.Random(f"{lang}-{seed}-C"), random.Random(f"{lang}-{seed}-D")   # separate streams: the main stream (and ids) stay stable
    used_en, used_pool = collections.Counter(), collections.Counter()
    used = collections.Counter()
    seen_ids: set = set()
    pool_cache: dict = {}
    genres = _quota(rnd, GENRES, n)
    lengths = ["brief"] * n if (lang in BRIEF or p.tier == "C") else _quota(rnd, LENGTHS, n)   # weakest languages: 1-3 short sentences only; others 2-5 / 6-12 / 13-20
    rL = random.Random(f"{lang}-{seed}-L")                                # own stream: the requested sentence count never shifts the other draws
    domains = _quota(rnd, {d: 1 for d in DOMAINS}, n)
    swt = {s: (3 if i == 0 else 1.5) for i, s in enumerate(p.switch)} if p.code != "pcm" else {s: 1 for s in p.switch}   # Pidgin: Yoruba/Igbo/Hausa equally, no English pin
    switches = _quota(rnd, swt, n)
    dwt = {d: (3 if i == 0 else 2) for i, d in enumerate(p.dialects)}
    dialects = _quota(rnd, dwt, n)
    personas, registers, emotions, media, audiences = (_quota(rnd, {x: 1 for x in L}, n) for L in (PERSONAS, REGISTERS, EMOTIONS, MEDIA, AUDIENCES))
    # long rows need coherent-piece genres
    plan = []
    for i in range(n):
        genre, length = genres[i], lengths[i]
        if length == "long" and genre in ("instruct", "market_bargain", "faith", "humor"):
            genre = rnd.choice(["story", "explain_teach", "chat", "opinion"])
        compat = COMPAT[genre]
        pool = [s for s in nat if s["genre"] in compat]
        track, seed_row = "A", None
        if genre == "story" and eng_stories and rnd.random() < 0.40:
            track, pool = "C", eng_stories
        elif pool and rnd.random() < 0.5:
            track = "B"
        if track != "A":
            low = min(used[s["seed_id"]] for s in pool)
            if low >= MAX_SEED_REUSE:          # pool exhausted: this row is anchor-seeded instead
                track = "A"
            else:
                seed_row = rnd.choice([s for s in pool if used[s["seed_id"]] == low])
                used[seed_row["seed_id"]] += 1
        domain, given = domains[i], None
        if track == "A" and en_dom.get(domain) and rC.random() < 0.35:        # anchors are the base; English domain seeds fill money/law/health/tech
            pool_d = [s for s in en_dom[domain] if (s["genre"] == "story") == (genre == "story")] or en_dom[domain]; low = min(used_en[s["seed_id"]] for s in pool_d)
            if low < MAX_SEED_REUSE:
                seed_row = rC.choice([s for s in pool_d if used_en[s["seed_id"]] == low]); used_en[seed_row["seed_id"]] += 1; track = "C"
        def _ns(t): return len(re.findall(r"[.!?]+(?:\s|$)", t))
        pool_k = pool_cache.get(length == "brief")                           # same filter as before, computed once per class (it re-ran a regex over the whole pool for every row)
        if pool_k is None:
            pool_k = pool_cache[length == "brief"] = [e for e in eng_pool if (_ns(e["eng"]) <= 3 if length == "brief" else (e.get("kind") != "sentence" and _ns(e["eng"]) >= 2))]
        if pool_k and rD.random() < REUSE_SHARE.get(lang, 0):                  # Track D: given English from another language's generated row
            low = min(used_pool[e["id"]] for e in pool_k)
            if low < 2:
                given = rD.choice([e for e in pool_k if used_pool[e["id"]] == low]); used_pool[given["id"]] += 1; track = "D"; seed_row = None
                if length != "brief":
                    ns = _ns(given["eng"]); length = "short" if ns <= 5 else ("medium" if ns <= 12 else "long")
                if given.get("genre") in GENRES and length != "brief": genre = given["genre"]
                if given.get("domain") in DOMAINS: domain = given["domain"]
        lo, hi = SENT_RANGE[length]
        n_sent = max(lo, min(hi, _ns(given["eng"]))) if given else rL.randint(lo, hi)     # exact requested sentence count (inside the band); given English keeps its own
        c = {"lang": lang, "track": track, "genre": genre, "length": length, "n_sent": n_sent, "domain": domain, "persona": personas[i],
             "dialect": dialects[i], "audience": audiences[i], "register": registers[i], "emotion": emotions[i], "medium": media[i], "switch": switches[i],
             "anchors": rnd.sample(anchors, min(len(anchors), rnd.choice([1, 2, 3]))) if (track == "A" and anchors) else [],
             "seed": seed_row["text"] if seed_row else "", "seed_id": seed_row["seed_id"] if seed_row else "",
             "seed_src": seed_row["src"] if seed_row else "",
             "given_eng": given["eng"] if given else "", "source_row": given["id"] if given else ""}
        def _id(c):
            return f"{lang}-{hashlib.md5(json.dumps({k: v for k, v in c.items() if k != 'id'}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]}"
        c["id"] = _id(c)
        for _ in range(20):                                   # uniqueness: a collision redraws anchors / persona until the card is new
            if c["id"] not in seen_ids:
                break
            if c["anchors"]:
                c["anchors"] = rnd.sample(anchors, min(len(anchors), rnd.choice([1, 2, 3])))
            c["persona"], c["audience"] = rnd.choice(PERSONAS), rnd.choice(AUDIENCES)
            c["id"] = _id(c)
        seen_ids.add(c["id"])
        plan.append(c)
    return plan


# ---------------- stage 3: generate ----------------
def do_generate(lang: str, n: int, model: str, conc: int, fewshot: bool, limit: int = 0):
    p, path = P[lang], OUT / f"{lang}.raw.jsonl"
    done = {r["id"] for r in jl(path)}
    plan = make_plan(lang, n)
    plan = [c for c in (plan[:limit] if limit else plan) if c["id"] not in done]
    pv = prov(model, conc)
    global GOLD
    GOLD = prompts.load_gold()

    def one(c):
        r = pv.complete(Request(c["id"], prompts.gen_prompt(p, c, GOLD), max_tokens=LENGTHS[c["length"]][2], temperature=GEN_TEMP, top_p=GEN_TOP_P,
                                response_format={"type": "json_object"}))
        d = prompts.parse_json(r.text) if r.ok else None
        row = {"id": c["id"], "lang": lang, "scenario": {**{k: c[k] for k in ("track", "genre", "length", "n_sent", "domain", "persona", "dialect", "audience", "register", "emotion", "medium", "switch")}, "length_spec": "v2"},
               "anchors": [a["term"] for a in c["anchors"]], "seed_id": c["seed_id"], "seed_src": c["seed_src"], "source_row": c.get("source_row", ""),
               "teacher": model, "prompt_fp": PROMPT_FP, "usage": [r.prompt_tokens, r.completion_tokens]}
        if not d:
            row["error"] = r.error or "bad_json"
            append(OUT / f"{lang}.errors.jsonl", row)
            return row
        row.update({k: d.get(k) for k in ("frame", "eng_initial", "eng", "m_text", "hm", "ood", "ood_relation", "devices", "self_check", "conf", "flags", "difficulty")})
        if c.get("given_eng"):
            row["eng"] = c["given_eng"]                        # track D: eng is given, never re-written
        append(path, row)
        return row
    with ThreadPoolExecutor(conc) as ex:
        rows = [f.result() for f in as_completed([ex.submit(one, c) for c in plan])]
    ok = [r for r in rows if "error" not in r]
    print(f"{lang}: generated {len(ok)}/{len(plan)}  tokens in/out {sum(r['usage'][0] for r in rows)}/{sum(r['usage'][1] for r in rows)}")


# ---------------- stage 4: gate + judge ----------------
def score(j: dict, g: dict, conf: dict) -> float:
    s = [v for v in (j.get("scores") or {}).values() if isinstance(v, (int, float))]
    sc = (sum(s) / len(s) / 5) if s else 0.0
    sc *= (j.get("certain") or 0.5) ** 0.25                 # judge-confidence discount
    sc -= 0.05 * len(g["warn"])
    gc = min([v for v in (conf or {}).values() if isinstance(v, (int, float))] or [0.5])
    return round(max(0.0, min(1.0, 0.8 * sc + 0.2 * gc)), 3)


def do_judge(lang: str, model: str, conc: int, tag: str):
    p = P[lang]
    raw, path = jl(OUT / f"{lang}.raw.jsonl"), OUT / f"{lang}.judged.{tag}.jsonl"
    done = {r["id"] for r in jl(path)}
    pv = prov(model, conc)

    def one(row):
        g = gates.check(row, p)
        out = dict(row)
        out["gates"] = g
        if g["fail"] or "seed_unusable" in (row.get("flags") or []):
            out["judge"], out["keep"], out["final_score"] = None, False, 0.0
            append(path, out)
            return out
        r = pv.complete(Request(row["id"], prompts.judge_prompt(p, row), max_tokens=LENGTHS[row["scenario"]["length"]][2] + 1500, temperature=0.2,
                                response_format={"type": "json_object"}))
        j = prompts.parse_json(r.text) if r.ok else None
        if not j:
            out["judge"], out["keep"], out["final_score"], out["judge_error"] = None, False, 0.0, r.error or "bad_json"
        else:
            out["judge"], out["judge_model"] = j, model
            cor = j.get("corrected") or {}
            out["corrected"] = {k: cor.get(k) for k in ("m_text", "hm", "ood") if cor.get(k) and cor[k] != row.get(k)}
            fin = {**row, **out["corrected"]}
            g2 = gates.check({**fin, "scenario": row["scenario"]}, p)
            out["gates_after"] = g2
            out["keep"] = bool(j.get("keep")) and not g2["fail"]
            out["final_score"] = score(j, g2, row.get("conf"))
            out["usage_judge"] = [r.prompt_tokens, r.completion_tokens]
        append(path, out)
        return out
    todo = [r for r in raw if r["id"] not in done]
    with ThreadPoolExecutor(conc) as ex:
        res = [f.result() for f in as_completed([ex.submit(one, r) for r in todo])]
    print(f"{lang}: judged {len(res)}  kept {sum(r['keep'] for r in res)}  gate-failed {sum(1 for r in res if r['gates']['fail'])}")


def do_adjudicate(lang: str, model: str, conc: int, tag: str, thr: float):
    """Blind second-model vote on every field the judge edited; writes final.<lang>.jsonl (+ preference pairs)."""
    p = P[lang]
    rows, path = jl(OUT / f"{lang}.judged.{tag}.jsonl"), OUT / f"{lang}.final.{tag}.jsonl"
    done = {r["id"] for r in jl(path)}
    pv = prov(model, conc)

    def one(r):
        rnd = random.Random(r["id"])
        out = {k: v for k, v in r.items() if k not in ("judge", "gates_after")}
        out["judge_scores"] = (r.get("judge") or {}).get("scores")
        out["judge_certain"] = (r.get("judge") or {}).get("certain")
        cor = r.get("corrected") or {}
        final = {k: r.get(k) for k in ("m_text", "hm", "ood")}
        prefs, votes = [], {}
        if cor and r.get("judge"):
            flip, pairs = {}, {}
            for f, new in cor.items():
                flip[f] = rnd.random() < 0.5
                pairs[f] = (new, r[f]) if flip[f] else (r[f], new)
            resp = pv.complete(Request(r["id"], prompts.adjudicate_prompt(p, r, pairs), max_tokens=int(os.environ.get("CS_ADJ_TOK", 1500)), temperature=0.0,
                                       response_format={"type": "json_object"}))
            d = prompts.parse_json(resp.text) if resp.ok else {}
            for f, new in cor.items():
                v = (d or {}).get(f)
                votes[f] = v
                new_is_x = flip[f]
                if v in ("X", "Y") and ((v == "X") == new_is_x):          # corrected version won
                    prefs.append({"field": f, "chosen": new, "rejected": r[f]})
                    final[f] = new
                elif v in ("X", "Y"):                                       # original won
                    prefs.append({"field": f, "chosen": r[f], "rejected": new})
        out["final"], out["adjudication"], out["prefs"] = final, votes, prefs
        fin_row = {**r, **final}
        g = gates.check({**fin_row, "scenario": r["scenario"]}, p)
        out["gates_final"] = g
        sc = r.get("final_score", 0.0)
        agree = [1 if (r.get("corrected") or {}).get(f) and votes.get(f) else 0 for f in votes]
        out["final_score"] = round(max(0.0, sc - 0.03 * sum(1 for f in votes if votes[f] is None)), 3)
        out["keep"] = bool(r.get("keep")) and not g["fail"] and out["final_score"] >= thr
        append(path, out)
        return out
    todo = [r for r in rows if r["id"] not in done]
    with ThreadPoolExecutor(conc) as ex:
        res = [f.result() for f in as_completed([ex.submit(one, r) for r in todo])]
    n_edit = sum(len(r["adjudication"]) for r in res)
    n_acc = sum(1 for r in res for p_ in r["prefs"] if p_["chosen"] != "" and p_["chosen"] == r["final"][p_["field"]] and (r["corrected"] or {}).get(p_["field"]) == p_["chosen"])
    print(f"{lang}: adjudicated {len(res)} rows; {n_edit} edits proposed, {n_acc} accepted; kept {sum(r['keep'] for r in res)}")


def _shingles(t: str, k: int = 4) -> set:
    w = re.findall(r"\w+", t.lower())
    return {hash(" ".join(w[i:i + k])) for i in range(max(1, len(w) - k + 1))}


def do_dedupe(lang: str, src_suffix: str, thr: float):
    """Uniqueness: drop a row whose `eng` (or `hm`) is a near-duplicate (4-gram shingle Jaccard >= thr) of an earlier kept row.
    Inverted index over shingles, so 6k rows/language stays fast. Writes <lang>.unique.jsonl and prints the drop rate."""
    rows = jl(OUT / f"{lang}.{src_suffix}.jsonl")
    kept, index, dropped = [], collections.defaultdict(list), collections.Counter()
    for r in rows:
        if not r.get("eng") or not r.get("hm"):
            dropped["empty"] += 1
            continue
        sh_e, sh_h = _shingles(r["eng"]), _shingles(r["hm"])
        cand = collections.Counter(j for s in sh_e for j in index[s])
        dup = next((j for j, n in cand.most_common(3) if n / len(sh_e | kept[j][1]) >= thr), None)
        if dup is not None:
            dropped["near_duplicate_eng"] += 1
            continue
        if any(r["hm"].strip() == k[0]["hm"].strip() for k in kept[-50:]):
            dropped["same_hm"] += 1
            continue
        i = len(kept)
        kept.append((r, sh_e))
        for s in sh_e:
            index[s].append(i)
    (OUT / f"{lang}.unique.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r, _ in kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"{lang}: {len(rows)} -> {len(kept)} unique ({dict(dropped)})")


def do_genall(langs: list, n: int, model: str, conc: int, limit: int, frac: float = 0.0):
    """One shared worker pool over ALL languages (interleaved), so the server never idles on a language's long-row tail.
    Same outputs / resume behaviour as do_generate: <lang>.raw.jsonl and <lang>.errors.jsonl."""
    global GOLD
    GOLD = prompts.load_gold()
    pv = prov(model, conc)
    todo, per_done = [], {}
    # per-language targets come from profiles.TARGETS (n=0); an explicit --n overrides (previews). frac = share of each language's target this round
    langs = [l for l in langs if (n or TARGETS.get(l, 0)) > 0]
    for lang in langs:
        nl = n or TARGETS[lang]
        path = OUT / f"{lang}.raw.jsonl"
        done = {r["id"] for r in jl(path)}
        plan = make_plan(lang, nl)
        if os.environ.get("CS_NO_LONG"):
            plan = [x for x in plan if x["length"] != "long"]         # previews: keep short/medium so the tail does not hold the GPU
        per_done[lang] = len(done)
        lim = int(round(frac * nl)) if frac else (limit or 0)
        todo.append([c for c in (plan[:lim] if lim else plan) if c["id"] not in done])
    # Prefix-cache locality: every language's system message has up to 3 variants (~8k tokens each). Rows are sorted by variant, cut into chunks that share one
    # system prefix, and the chunks are interleaved across languages - so every language still advances together, but the ~48 requests in flight share
    # 1-2 prefixes instead of 40 (KV memory and prefill drop by several times).
    CH = int(os.environ.get("CS_CHUNK", 24))
    chunks = []
    for lst in todo:
        lst = sorted(lst, key=lambda c: (prompts.variant_for(c["lang"], c["id"]), c["length"]))
        chunks.append([lst[i:i + CH] for i in range(0, len(lst), CH)])
    items = [x for grp in zip_longest_interleave(chunks) for ch in grp for x in ch]
    tok = collections.Counter()

    def one(c):
        lang, p = c["lang"], P[c["lang"]]
        r = pv.complete(Request(c["id"], prompts.gen_prompt(p, c, GOLD), max_tokens=LENGTHS[c["length"]][2], temperature=GEN_TEMP, top_p=GEN_TOP_P,
                                response_format={"type": "json_object"}))
        d = prompts.parse_json(r.text) if r.ok else None
        row = {"id": c["id"], "lang": lang, "scenario": {**{k: c[k] for k in ("track", "genre", "length", "n_sent", "domain", "persona", "dialect", "audience", "register", "emotion", "medium", "switch")}, "length_spec": "v2"},
               "anchors": [a["term"] for a in c["anchors"]], "seed_id": c["seed_id"], "seed_src": c["seed_src"], "source_row": c.get("source_row", ""), "teacher": model, "prompt_fp": PROMPT_FP, "usage": [r.prompt_tokens, r.completion_tokens]}
        with _lock:
            tok["in"] += r.prompt_tokens; tok["out"] += r.completion_tokens; tok["rows"] += 1
        if not d:
            row["error"] = r.error or "bad_json"
            append(OUT / f"{lang}.errors.jsonl", row)
            return
        row.update({k: d.get(k) for k in ("frame", "eng_initial", "eng", "m_text", "hm", "ood", "ood_relation", "devices", "self_check", "conf", "flags", "difficulty")})
        if c.get("given_eng"):
            row["eng"] = c["given_eng"]                        # track D: eng is given, never re-written
        append(OUT / f"{lang}.raw.jsonl", row)
    print(f"genall: {len(items)} rows to do over {len(langs)} languages, {conc} workers", flush=True)
    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, items))
    print(f"genall done: {tok['rows']} rows, tokens in/out {tok['in']}/{tok['out']}", flush=True)


def zip_longest_interleave(lists):
    """round-robin over the per-language lists so every language advances together"""
    import itertools
    for grp in itertools.zip_longest(*lists):
        yield [x for x in grp if x is not None]


def do_rehm(lang: str, model: str, conc: int, limit: int):
    """Regenerate ONLY `hm` for rows that already have eng + m_text, with the current prompt wording. Writes <lang>.rehm.jsonl."""
    global GOLD
    GOLD = prompts.load_gold()
    p, rows = P[lang], jl(OUT / f"{lang}.raw.jsonl")
    done = {r["id"] for r in jl(OUT / f"{lang}.rehm.jsonl")}
    plan = {c["id"]: c for c in make_plan(lang, 6000)}
    todo = [r for r in rows if r.get("hm") and r["id"] not in done and r["id"] in plan][: limit or None]
    pv = prov(model, conc)

    def one(r):
        c = plan[r["id"]]
        card = prompts.gen_prompt(p, c, GOLD)
        user = card[1]["content"].split("\nHOW TO START")[0] + (
            "\n\nThe plain English and the official translation of this sample are already written:\n"
            f"ENG: {r['eng']}\nM_TEXT: {r['m_text']}\n\n"
            'Write ONLY the hm field (item 3) for this sample: a code-switched variation of M_TEXT. Return exactly {"hm": "..."}')
        resp = pv.complete(Request(r["id"], [card[0], {"role": "user", "content": user}], max_tokens=1500, temperature=0.85,
                                   response_format={"type": "json_object"}))
        d = prompts.parse_json(resp.text) if resp.ok else None
        if d and d.get("hm"):
            append(OUT / f"{lang}.rehm.jsonl", {"id": r["id"], "lang": lang, "hm_old": r["hm"], "hm_new": d["hm"], "prompt_fp": PROMPT_FP, "usage": [resp.prompt_tokens, resp.completion_tokens]})
    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, todo))
    print(f"{lang}: rehm {len(jl(OUT / f'{lang}.rehm.jsonl'))} rows")


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, 0-100), pure python so it runs on any box."""
    h, r = re.sub(r"\s+", "", hyp.lower()), re.sub(r"\s+", "", ref.lower())
    ps, rs = [], []
    for k in range(1, n + 1):
        hc = collections.Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = collections.Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        ov = sum((hc & rc).values())
        ps.append(ov / sum(hc.values())); rs.append(ov / sum(rc.values()))
    if not ps:
        return 0.0
    p, rr = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + rr == 0 else 100 * (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


BT_SYS = "You are a precise translator. Translate the user's text into plain, literal English. Output ONLY the English translation."
MT_SYS = ("You are a professional translator and native speaker. Translate the user's plain English into {name} ({dialect} not needed: use the standard variety). "
          "Faithful and complete - every fact, number and nuance - in natural, grammatical, standard-register {name}. Orthography: {orth} Output ONLY the translation.")


def do_mcheck(lang: str, model: str, conc: int, limit: int, fix: bool, thr: float, ncand: int = 4):
    """Accuracy gate for m_text, for EVERY language: back-translate m_text to English (the model never sees eng), score chrF vs eng.
    With fix=True, rows below `thr` get `ncand` fresh translations of eng; the best by back-translation chrF replaces m_text if it is
    better (old one is kept as m_text_rejected). Writes <lang>.mcheck.jsonl."""
    p = P[lang]
    rows = jl(OUT / f"{lang}.raw.jsonl")
    done = {r["id"] for r in jl(OUT / f"{lang}.mcheck.jsonl")}
    todo = [r for r in rows if r.get("m_text") and r.get("eng") and r["id"] not in done][: limit or None]
    pv = prov(model, conc)

    def bt(text: str) -> str:
        r = pv.complete(Request("bt", [{"role": "system", "content": BT_SYS}, {"role": "user", "content": text}], max_tokens=2500, temperature=0.0))
        return (r.text or "").strip()

    def one(r):
        b = bt(r["m_text"]); s0 = chrf(b, r["eng"])
        out = {"id": r["id"], "lang": lang, "bt": b, "chrf_old": round(s0, 1), "m_text_old": r["m_text"], "m_text_new": None, "chrf_new": None, "status": "ok" if s0 >= thr else "low"}
        if fix and s0 < thr:
            best, bs = None, s0
            for _ in range(ncand):
                rq = pv.complete(Request("mt", [{"role": "system", "content": MT_SYS.format(name=p.name, dialect=r["scenario"]["dialect"], orth=p.orthography)},
                                                 {"role": "user", "content": r["eng"]}], max_tokens=LENGTHS[r["scenario"]["length"]][2], temperature=0.6))
                cand = (rq.text or "").strip()
                if not cand:
                    continue
                sc = chrf(bt(cand), r["eng"])
                if sc > bs:
                    best, bs = cand, sc
            if best and bs >= s0 + 5:
                out.update(m_text_new=best, chrf_new=round(bs, 1), status="fixed" if bs >= thr else "improved_below_threshold")
            else:
                out["status"] = "unverified"
        append(OUT / f"{lang}.mcheck.jsonl", out)
    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, todo))
    res = jl(OUT / f"{lang}.mcheck.jsonl")
    cnt = collections.Counter(x["status"] for x in res)
    print(f"{lang}: m_text check {len(res)} rows {dict(cnt)}; chrF mean old {sum(x['chrf_old'] for x in res)/max(1,len(res)):.1f}", flush=True)


def do_report(lang: str, tag: str):
    rows = jl(OUT / f"{lang}.judged.{tag}.jsonl")
    if not rows:
        return
    import statistics as st
    sc = [r["final_score"] for r in rows if r.get("judge")]
    dims = {}
    for r in rows:
        for k, v in ((r.get("judge") or {}).get("scores") or {}).items():
            dims.setdefault(k, []).append(v)
    print(f"== {lang}: n={len(rows)} kept={sum(r['keep'] for r in rows)} mean_score={st.mean(sc) if sc else 0:.2f}")
    print("   " + "  ".join(f"{k}={st.mean(v):.2f}" for k, v in dims.items()))
    fails = {}
    for r in rows:
        for f in r["gates"]["fail"] + r["gates"]["warn"]:
            fails[f] = fails.get(f, 0) + 1
    print("   gates:", fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["anchors", "seeds", "generate", "judge", "adjudicate", "dedupe", "report", "plan", "genall", "rehm", "mcheck"])
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default=os.environ.get("CSW_GEN_MODEL", ""))
    ap.add_argument("--judge-model", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--adj-model", default=os.environ.get("CSW_ADJ_MODEL", ""))
    ap.add_argument("--thr", type=float, default=0.60)
    ap.add_argument("--tag", default="g")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--per-cat", type=int, default=40)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--fix", action="store_true", help="mcheck: regenerate low-scoring m_text (best of 4 by back-translation)")
    ap.add_argument("--thr-m", type=float, default=45.0, help="mcheck: chrF(back-translation, eng) below this = suspect")
    ap.add_argument("--frac", type=float, default=0.0, help="genall: this round generates FRAC of every language's target (profiles.TARGETS)")
    ap.add_argument("--limit", type=int, default=0, help="generate only the first LIMIT rows of the --n plan (resumable: raise it later)")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--no-nc", action="store_true")
    ap.add_argument("--no-fewshot", action="store_true")
    a = ap.parse_args()
    global PROVIDER
    PROVIDER = a.provider
    if a.stage in ("generate", "genall", "rehm", "mcheck", "judge", "adjudicate"):
        try:                                   # keep prompt_preview.md in step with the prompts that are about to run
            from csw.datagen import render_preview
            global PROMPT_FP
            PROMPT_FP = render_preview.render()[0] + (('-' + prompts.VARIANT_TAG) if prompts.VARIANT_TAG else '')      # variant (rules-first / shots) is part of the fingerprint stored on every row
            print("prompt preview fingerprint", PROMPT_FP, flush=True)
        except Exception as e:
            print("prompt preview not refreshed:", e, flush=True)
    if a.stage == "genall":
        do_genall(a.langs, a.n, a.model, a.conc, a.limit, a.frac)
        return
    for lg in a.langs:
        if lg not in P:
            raise SystemExit(f"{lg}: not in profiles {sorted(P)}")
        if a.stage == "anchors":
            do_anchors(lg, a.model, a.per_cat, a.conc, a.round)
        elif a.stage == "seeds":
            r = seedmod.build(lg, allow_nc=not a.no_nc)
            print(lg, "seeds:", len(r))
        elif a.stage == "plan":
            pl = make_plan(lg, a.n)
            print(lg, {t: sum(1 for c in pl if c["track"] == t) for t in "ABC"})
        elif a.stage == "generate":
            do_generate(lg, a.n, a.model, a.conc, not a.no_fewshot, a.limit)
        elif a.stage == "judge":
            do_judge(lg, a.judge_model, a.conc, a.tag)
        elif a.stage == "mcheck":
            do_mcheck(lg, a.model, a.conc, a.limit, a.fix, a.thr_m)
        elif a.stage == "rehm":
            do_rehm(lg, a.model, a.conc, a.limit)
        elif a.stage == "dedupe":
            do_dedupe(lg, "raw", 0.5)
        elif a.stage == "adjudicate":
            do_adjudicate(lg, a.adj_model, a.conc, a.tag, a.thr)
        elif a.stage == "report":
            do_report(lg, a.tag)


if __name__ == "__main__":
    main()
