"""Ask a strong model (default google/gemini-3.1-pro-preview via OpenRouter) to SCORE, give a VERDICT on and CORRECT gold rows, and to say which languages it is not good at.
Proposals only - nothing touches augmented_gold.jsonl.
python -m csw.judging.repair_gold <rows.jsonl | --all> [--model M] [--limit N] [--lang efi] [--dry] [--out file] [--conc 4] [--effort high|medium]
  --all                     every gold row (augmented_gold.jsonl + gold.jsonl); rows listed in exemplars/review_round2.jsonl carry their flag reason
  --batch submit|poll       OpenRouter batch API (50% price, up to 24h): submit writes <out>.batch.json, poll writes the results into <out>
  --dry                     render the first prompt + cost estimate, no API call
Each output row = the original row + `proposal` {verdict, scores_before, scores_after, eng, m_text, hm, ood, changes, language_proficiency, notes} + `usage`."""
from csw.paths import DATA_ROOT, env_value
import json, os, re, sys, time, urllib.request, urllib.error, concurrent.futures as cf
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen import cs_prompts as pr
from csw.datagen import seeds as seedmod
from csw.datagen import gates
from csw.datagen.profiles import P

PRICE = {"google/gemini-4-argon": (2e-6, 1e-5), "google/gemini-3.1-pro-preview": (2e-6, 1.2e-5)}   # realtime; batch is 50%   # $ / token (prompt, completion)
SYSTEM = """You are a panel of expert native speakers, professional translators and linguists auditing a gold-standard dataset of how African languages are really written and spoken. Dataset rows were corrected by hand; some were also flagged by automatic checks. For every row you (1) SCORE it, (2) give a VERDICT, (3) CORRECT what is wrong, and (4) honestly report how well you know the language - with the highest possible accuracy.

THE FOUR FIELDS (what each must be)
1. eng - FORMAL, NEUTRAL English: the official English statement of the content. No emotive words, idioms, metaphors, slang, contractions, exclamations or cultural colour; complete explicit sentences; same facts and speaker's point.
2. m_text - the OFFICIAL translation of eng into the target (base) language: faithful, grammatical, natural, standard orthography and standard dialect, as formal and plain as eng. It must be real text of the target language - never another language.
3. hm (human-mixed) - a natural code-switched variation of m_text, as a native speaker really talks (WhatsApp, voice note, phone call). The TARGET language is the base (matrix) language and must clearly lead; only the PINNED SWITCH LANGUAGE supplies words and short phrases (mostly 1-4 words, plus about one longer phrase of up to ~10 words) at natural switch points. About 15-40% of the words come from the switch language (a third on average, never more than about 40%). The base language keeps its own grammatical frame (verbs, tense/aspect markers, pronouns, particles, negation, sentence structure). No third language. Full diacritics on every word of every language that has them. If the pinned switch language is English or Pidgin, use only the informal everyday words natives really drop in (app, scam, stress, on time, network), never formal or bookish English; Pidgin words must be real Pidgin (no English past tense, copulas or auxiliaries).
4. ood - the complete, informal, native parlance of the same situation: rich in idiom, proverbs, imagery, ideophones, praise/oath formulae and native sentence structures; same situation, facts and intent as eng; dominantly native and clearly richer than hm; may keep a few nativised loans or light code-switching exactly as natives do.

RULES
- Never invent words, forms, compounds or proverbs. If you are not certain a word exists in the language, use a simpler native construction you are certain of.
- Use the full standard orthography including every diacritic / special letter of the language (see ORTHOGRAPHY in the task). Never ASCII stand-ins.
- Never insult; no slurs.
- Each field must be genuinely in ITS language: a field copied or lightly edited from a different language (even a related one) is wrong and must be rewritten in the target language from eng.
- Keep what is already correct. Change a field only when you can say what was wrong with it. If eng itself is not formal/neutral, or its facts do not match, repair eng first and say so.
- If you cannot write the target language reliably for a field, do NOT guess: set verdict to "unsalvageable" for the row (or keep the field and list it in notes) and lower language_confidence. An honest "I am not sure" is far more useful than fluent-looking invented text.

HOW TO WORK: (1) read the flag reason and the row; (2) compare every field with eng and with the reference material; (3) decide for each field: correct / wrong-language / mistranslated / over-mixed / under-native / diacritics; (4) rewrite only what is wrong, word by word; (5) re-read the final row once more against the rules.

SCORING (1-10 per field, applied to the row AS GIVEN, before your changes): 10 = a native expert would publish it unchanged; 8 = good, minor slips; 6 = understandable but with clear errors (wrong words, over/under-mixing, missing diacritics); 4 = mostly wrong or partly in another language; 1-2 = wrong language, copied from another language, or meaningless. eng: formal, neutral and complete. m_text: faithful, grammatical, real target language. hm: pin discipline, 15-40% switch, base leads, natural switch points. ood: native richness, idiom, correct language. Then score your corrected version again in scores_after.

VERDICT: "ok" (nothing needs changing), "fixed" (you corrected it and are confident the corrected version is accurate), "partial" (you corrected some fields; others you could not verify), "unsalvageable" (you cannot reliably write this language for this row - say so; do not fabricate).

LANGUAGE_PROFICIENCY - be honest, this decides which languages get human reviewers: level = "fluent" (you write it like a native), "good" (reliable apart from rare words), "limited" (you recognise it and can fix obvious errors but may invent or mis-spell words), "poor" (you cannot verify or write it); can_write_reliably = true/false; weak_areas = a short note (for example "orthography and tone marks", "ideophones and proverbs", "dialect X", "vocabulary of the register"). The same applies to the pinned switch language if it is not English.

Return ONE JSON object and nothing else:
{"verdict": "ok" | "fixed" | "partial" | "unsalvageable",
 "scores_before": {"eng": 0, "m_text": 0, "hm": 0, "ood": 0},
 "eng": "...", "m_text": "...", "hm": "...", "ood": "...",
 "scores_after": {"eng": 0, "m_text": 0, "hm": 0, "ood": 0},
 "changes": [{"field": "m_text|hm|ood|eng", "problem": "what was wrong, in English", "fix": "what you changed"}],
 "language_proficiency": {"language": "...", "level": "fluent|good|limited|poor", "can_write_reliably": true, "weak_areas": "..."},
 "notes": "anything the human reviewer must double-check (uncertain words, dialect, diacritics)"}
Always return all four text fields (unchanged ones copied exactly)."""


def _read(path):
    t, dec, i, out = Path(path).read_text(encoding="utf-8"), json.JSONDecoder(), 0, []
    while True:
        while i < len(t) and t[i].isspace(): i += 1
        if i >= len(t): return out
        o, i = dec.raw_decode(t, i); out.append(o)


def card(row: dict, aug: dict, flagged_keys: set) -> str:
    p = P[row["lang"]]
    sw = row["switch"]
    own = [r for r in aug.get(row["lang"], []) if pr._key(r) not in flagged_keys and r["eng"] != row["eng"]][:5]
    out = [f"TARGET (base) LANGUAGE: {p.name} [{p.code}]\nPINNED SWITCH LANGUAGE for hm: {sw}{pr._switch_orth(sw)}\nDIALECT / VARIETY: {row.get('dialect') or 'standard'}\nGENRE: {row['genre']}\n"
           f"ORTHOGRAPHY: {p.orthography}\nSTREET / VERNACULAR REGISTER (for ood): {p.street}\n" + (f"LANGUAGE NOTE: {p.extra}\n" if p.extra else "")
           + f"ALL SWITCH LANGUAGES ATTESTED FOR {p.name}: {', '.join(p.switch)}"]
    out.append(f"\nAUTOMATIC FLAG (the checks may be wrong - verify): {row['flag_reason']}" if row.get("flag_reason") else "\nAUTOMATIC FLAG: none - routine audit of a hand-corrected row; look for errors anyway.")
    out.append("\nTHE ROW TO AUDIT AND REPAIR:\n" + json.dumps({k: row[k] for k in ("eng", "m_text", "hm", "ood") if k in row}, ensure_ascii=False, indent=1))
    partners = re.findall(r"with the (\w+) row \((\w+)\)", row.get("flag_reason", ""))
    for pl_, pg in list(dict.fromkeys(partners))[:3]:
        for r in aug.get(pl_, []):
            if r["genre"] == pg:
                out.append(f"\nCOPY SUSPECT - the row in {P[pl_].name if pl_ in P else pl_} that this one overlaps with (it may itself be the wrong one; do NOT write {P[pl_].name if pl_ in P else pl_}):\n"
                           + json.dumps({k: r[k] for k in ("eng", "m_text", "hm", "ood")}, ensure_ascii=False))
                break
    if own:
        out.append(f"\nOTHER NATIVE-CORRECTED {p.name.upper()} ROWS (reference for orthography, switch style and richness - never reuse their content. CAUTION: these were also hand-corrected and some contain fragments of other languages; where they disagree with the ATTESTED REAL TEXT below, the attested text wins):\n"
                   + "\n".join(json.dumps({k: r[k] for k in ("eng", "m_text", "hm", "ood")}, ensure_ascii=False) for r in own))
    sd = [s["text"] for s in seedmod.load(row["lang"]) if s["track"] == "B" and 20 <= len(s["text"].split()) <= 80][:6]
    if sd:
        out.append(f"\nATTESTED REAL {p.name.upper()} TEXT (genuine writing from corpora - authoritative for spelling, diacritics and everyday vocabulary):\n" + "\n".join("- " + s for s in sd))
    out.append("\nNow audit and repair the row. Return the JSON object.")
    return "\n".join(out)


EFFORT = "high"


def _key_env():
    return env_value("OPENROUTER_API_KEY")


def call(model, system, user, max_tokens=24000):
    key = env_value("OPENROUTER_API_KEY")
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": 0.3,
            "response_format": {"type": "json_object"}, "reasoning": {"effort": EFFORT}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    last = None
    for i in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=600))
        except Exception as e:
            last = e; time.sleep(5 * 2 ** i)
    raise RuntimeError(str(last))


def main():
    a = sys.argv[1:]
    src = a[0]; opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    model, limit, only = opt("--model", "google/gemini-3.1-pro-preview"), int(opt("--limit", 0)), opt("--lang")
    global EFFORT; EFFORT = opt("--effort", "high")
    if src == "--all":
        src = str(HERE / "exemplars" / "augmented_gold.jsonl"); allrows = _read(src) + _read(HERE / "exemplars" / "gold.jsonl")
        allrows = [r for r in allrows if "status" not in r]
        fl = {(r["lang"], r["eng"]): r["flag_reason"] for r in (_read(HERE / "exemplars" / "review_round2.jsonl") if (HERE / "exemplars" / "review_round2.jsonl").exists() else [])}
        for r in allrows:
            if (r["lang"], r["eng"]) in fl: r["flag_reason"] = fl[(r["lang"], r["eng"])]
        default_out = HERE / "exemplars" / "gold_audit.gemini.jsonl"
    else:
        allrows = _read(src); default_out = Path(src).with_suffix(".gemini.jsonl")
    out = Path(opt("--out", str(default_out)))
    rows = [r for r in allrows if not only or r["lang"] == only]
    rows = rows[:limit] if limit else rows
    aug = pr.load_aug_gold()
    flagged = {pr._key(r) for r in allrows if r.get("flag_reason")}
    if "--only-flagged" in a: rows = [r for r in rows if r.get("flag_reason")]
    if "--skip-flagged" in a: rows = [r for r in rows if not r.get("flag_reason")]
    done = {(r["lang"], r["eng"]) for r in _read(out)} if out.exists() else set()
    for f in a[a.index("--also-done") + 1:a.index("--also-done") + 2] if "--also-done" in a else []:
        done |= {(r["lang"], r["eng"]) for r in _read(f)}
    todo = [r for r in rows if (r["lang"], r["eng"]) not in done]
    if "--dry" in a:
        u = card(todo[0], aug, flagged); n_in = (len(SYSTEM) + len(u)) // 3
        pi, po = PRICE.get(model, (2e-6, 1e-5))
        print(u, "\n\n--- prompt ~", n_in, "tokens; estimate per row $%.3f (assuming ~6k output incl. reasoning); %d rows -> $%.2f" % (n_in * pi + 6000 * po, len(todo), len(todo) * (n_in * pi + 6000 * po)))
        return
    mode = opt("--batch")
    if mode:
        meta = Path(str(out) + ".batch.json"); H = {"Authorization": f"Bearer {_key_env()}", "Content-Type": "application/json"}
        def api(method, path, body=None):
            rq = urllib.request.Request("https://openrouter.ai/api/v1" + path, data=json.dumps(body).encode() if body else None, method=method, headers=H)
            try:
                return json.load(urllib.request.urlopen(rq, timeout=300))
            except urllib.error.HTTPError as e:
                return {"http_error": e.code, "body": e.read().decode()[:600]}
        if mode == "submit":
            bm = opt("--batch-model", model + ":batch")
            seen_x = set(); todo = [r for r in todo if not ((r["lang"], r["eng"], r["m_text"], r["hm"], r["ood"]) in seen_x or seen_x.add((r["lang"], r["eng"], r["m_text"], r["hm"], r["ood"])))]   # exact duplicates are audited once
            reqs = [{"custom_id": f"{r['lang']}-{pr._key(r)}-{i}", "body": {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": card(r, aug, flagged)}], "max_tokens": int(opt("--max-tokens", 12000)), "temperature": 0.3,
                     "response_format": {"type": "json_object"}, "reasoning": {"effort": EFFORT}}} for i, r in enumerate(todo)]
            d = api("POST", "/batches", {"endpoint": "/v1/chat/completions", "model": bm, "completion_window": "24h", "requests": reqs})
            print({k: d.get(k) for k in ("id", "status", "http_error", "body", "error")}); 
            if d.get("id"): meta.write_text(json.dumps({"id": d["id"], "model": bm, "rows": {f"{r['lang']}-{pr._key(r)}-{i}": r for i, r in enumerate(todo)}}, ensure_ascii=False))
            return
        if mode == "poll":
            m = json.loads(meta.read_text()); d = api("GET", "/batches/" + m["id"])
            print(d.get("status"), d.get("request_counts"), d.get("usage"), d.get("error") or d.get("http_error"))
            if d.get("results"):
                with out.open("a", encoding="utf-8") as fo:
                    for it in d["results"]:
                        r = m["rows"].get(it["custom_id"]); body = (it.get("response") or {}).get("body") or {}
                        msg = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                        rec = dict(r); rec["proposal"] = pr.parse_json(msg) or {"verdict": "error", "raw": msg[:500], "error": it.get("error")}; rec["model"] = m["model"]; rec["usage"] = body.get("usage")
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print("wrote", len(d["results"]), "->", out)
            return
    lock_out = out.open("a", encoding="utf-8")
    def one(r):
        t0 = time.time(); d = call(model, SYSTEM, card(r, aug, flagged)); msg = d["choices"][0]["message"].get("content") or ""
        prop = pr.parse_json(msg) or {"verdict": "error", "raw": msg[:500]}
        rec = dict(r); rec["proposal"] = prop; rec["model"] = model; rec["usage"] = d.get("usage"); rec["seconds"] = round(time.time() - t0)
        lock_out.write(json.dumps(rec, ensure_ascii=False) + "\n"); lock_out.flush()
        print(r["lang"], r["genre"], prop.get("verdict"), "cost", (d.get("usage") or {}).get("cost"), flush=True)
    with cf.ThreadPoolExecutor(int(opt("--conc", 4))) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
