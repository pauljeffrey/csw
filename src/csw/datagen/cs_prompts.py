"""Prompts for anchor mining, generation and judging."""
from __future__ import annotations

from csw.paths import DATA_ROOT
import json

from csw.datagen.profiles import GENRES, LENGTHS, Profile

ANCHOR_CATS = ["food and cooking", "kinship and address terms", "ceremonies and rites of passage", "markets, money and informal finance",
               "transport and daily commute", "religion and spirituality", "customary law, chieftaincy and governance", "folklore, tricksters and animals",
               "plants, animals and landscape", "music, dance, film and sport", "clothing, beauty and body", "health, healing and herbal medicine",
               "farming, fishing and herding", "school, work and migration", "politics, protest and civic life", "technology and youth life in this region",
               "humour, insults, praise and greetings", "philosophy, proverbs and moral concepts", "festivals, calendar and weather", "home, housing and domestic life"]

ANCHOR_SYS = "You are a native-speaker ethnographer and lexicographer. You never invent words; if unsure you omit the item."


def anchor_prompt(p: Profile, cat: str, n: int = 40) -> str:
    return (f"List {n} culturally specific concepts, objects, institutions, practices, sayings or terms from {p.name}-speaking life "
            f"(region: {', '.join(p.persona_regions)}) in the category \"{cat}\". Prefer things with NO tidy English equivalent or that an outsider would "
            f"not know. Return JSON only: {{\"items\":[{{\"term\":\"<in {p.name}, full orthography: {p.orthography}>\",\"gloss\":\"<short English gloss>\","
            f"\"untranslatable\":true|false}}]}}. No duplicates; no generic words like 'food' or 'family'.")


V2_RULES = """

QUALITY BAR (common failures to avoid):
1. Think and write in {name} first. Never produce English word order with {name} words sprinkled in; build every sentence on {name} grammar.
2. hm may contain ONLY {name} and {switch}. Do not leak any third language (e.g. Pidgin) unless {switch} is Pidgin. Switch whole words or phrases that a real speaker would really say in {switch} (tech, money, school, work, discourse markers like "actually/so/anyway", emphasis, quotes) - not rare words.
3. hm must differ visibly from m_text: different word order, native particles/interjections, contractions, drops and repetitions typical of speech. Never a word-for-word swap of m_text.
4. Do not translate English idioms literally. Express the idea with an idiom {name} speakers actually use. If you do not know a real idiom, use concrete native imagery instead of inventing a "proverb".
5. ood: at least two different devices (proverb, ideophone, praise/oath formula, parallelism, rhetorical question, vivid local imagery), concrete not abstract, native syntax, no foreign words. It may be longer than eng.
6. Diacritics/orthography exactly as specified in orth, in m_text, hm AND ood.
7. Dialect {dialect}: use its genuine forms only where they truly differ from standard; m_text stays standard."""


def _persona_block(c: dict) -> str:
    return (f"speaker: {c['persona']}; region/dialect: {c['dialect']}; register: {c['register']}; mood: {c['emotion']}; "
            f"medium: {c['medium']}")


SYSTEM = """You are three native {name} speakers working as one: (1) a professional translator, (2) a young adult who lives in {regions} and whose daily WhatsApp and voice notes mix {name} with {switches}, and (3) an elder-turned-street raconteur with deep idiom. You write the way real people talk, never the way textbooks or machine translation do. You never invent words or fake proverbs: if you are unsure of a form, choose a safer native construction and lower your confidence. Output is JSON only."""

FIELDS = """Produce these fields (all must express the SAME situation and core message; wording and structure may differ greatly):
- frame: 2-3 lines in English: speech act, who-to-whom, key entities, emotion, any local concepts/idioms you intend to use. (Plan before writing.)
- eng: the exact, natural English of the situation. It is the final English reference. Keep untranslatable local terms in the original form (do NOT explain them inside the sentence). No Western defaults (US tax, Thanksgiving, 401k...) unless the seed demands it.
- m_text: careful, standard, official-register {name} translation of eng. Natural, not calqued from English word order; uses correct standard orthography.
- hm (human-mixed): how a native speaker from {dialect} would actually say it in everyday chat/voice while code-switching with {switch}. Matrix language = {name}; {switch} supplies content words, technical terms, discourse markers, tags, quoted phrases or emphasis at GRAMMATICAL switch points (never random word swapping, never English word order with {name} words sprinkled in). About 15-45% of words from {switch}, varying by sentence. Real contractions, idioms, interjections, emotional exclamations, particles, and complex native structures. RESTRUCTURE; do not just substitute words in m_text. Switch ONLY into {switch}.
- ood (out-of-domain / deepest vernacular): the richest street-and-heritage rendering a deep native speaker could give: {street}. Dense with authentic idioms, proverbs, ideophones, praise/oath/blessing formulae, vivid imagery and native syntax. Same situation and communicative intent as eng, but free in expression (reorder, add apt imagery, longer or shorter by up to +-50%) without changing the facts or speech act. No raw foreign words (fully nativised loans are fine).
- orth: orthography rules to obey in m_text, hm and ood -> {orth}
- conf: your honest 0-1 confidence for m_text, hm and ood (lower it whenever you doubt a word, a dialect form or an idiom).
- ood_relation: "equivalent" | "idiomatic_paraphrase" | "proverb_equivalent" (how ood relates to eng).
- devices: list of idiom/proverb/ideophone devices actually used in ood (native form + short English gloss).
- flags: list from ["uncertain_terms","seed_unusable","politically_sensitive","pii_removed"] (empty if none)."""

OUT = '{"frame": "...", "eng_initial": "...", "eng": "...", "m_text": "...", "hm": "...", "ood": "...", "ood_relation": "...", "devices": [], "self_check": ["..."], "conf": {"m_text": 0.0, "hm": 0.0, "ood": 0.0}, "flags": [], "difficulty": 0}'


def gen_prompt_legacy(p: Profile, c: dict, fewshot: list[dict] | None = None, exemplars: list[dict] | None = None, v2: bool = False) -> list[dict]:
    sw = c["switch"]
    L = LENGTHS[c["length"]][1]
    system = SYSTEM.format(name=p.name, regions=", ".join(p.persona_regions), switches=" / ".join(p.switch))
    fields = FIELDS.format(name=p.name, dialect=c["dialect"], switch=sw, street=p.street, orth=p.orthography)
    if c["track"] == "A":
        head = (f"TASK (track A, anchor-seeded). First write the English situation yourself, then the {p.name} fields.\n"
                f"genre: {c['genre']} ({GENRES[c['genre']][1]})\ndomain: {c['domain']}\nlength of eng: {L}\n{_persona_block(c)}\n"
                f"must naturally involve these local concepts (keep their {p.name} form inside eng): "
                + "; ".join(f"{a['term']} ({a['gloss']})" for a in c["anchors"]) +
                f"\nThe English must sound like a thoughtful bilingual from {', '.join(p.persona_regions[:2])} speaking English, not a US/UK text.")
    elif c["track"] == "B":
        head = (f"TASK (track B, native-seeded). The seed below is real {p.name} text. Derive a NEW {c['genre']} situation inspired by its topic and voice "
                f"(do not translate it; do not copy private details), then write all fields. eng must be natural English for the situation you create; "
                f"length of eng: {L}\n{_persona_block(c)}\nSEED ({p.name}): \"\"\"{c['seed']}\"\"\"\n"
                f"If the seed is gibberish, wrong-language, or unusable, return flags:[\"seed_unusable\"] and empty strings.")
    else:
        head = (f"TASK (track C, English seed localisation). Adapt the English seed below into a {c['genre']} situation set in {c['dialect']} "
                f"({', '.join(p.persona_regions[:2])}) life: replace Western names, foods, institutions and settings with local ones, keep the narrative skeleton, "
                f"and set eng = the LOCALISED English. length of eng: {L}\n{_persona_block(c)}\nENGLISH SEED: \"\"\"{c['seed']}\"\"\"")
    fs = ""
    if fewshot:
        fs = ("\n\nShape reference only (real-looking mixed utterances; they are short and under-marked for diacritics, so do NOT copy their orthography or length):\n"
              + "\n".join(f"- {r['utterance_mixed']}" for r in fewshot))
    extra = ""
    if v2:
        extra += V2_RULES.format(name=p.name, switch=sw, dialect=c["dialect"])
    if exemplars:
        ex = "\n".join(json.dumps({k: e[k] for k in ("language", "eng", "m_text", "hm", "ood", "ood_relation", "devices") if k in e}, ensure_ascii=False) for e in exemplars)
        if any(e.get("language") and e["language"] != p.name for e in exemplars):
            ex += f"\n(Examples whose \"language\" is not {p.name} only show the DEPTH and STRUCTURE to reach; write {p.name}, never those languages, and use {p.name} idioms.)"
        extra += ("\n\nREFERENCE ANSWERS by expert native writers, for OTHER situations. Match their LEVEL: how far hm departs from m_text, how the switch language is woven in, how dense and concrete ood is, "
                  "and their diacritics. Do NOT reuse their content, names, proverbs or phrasing:\n" + ex)
    user = f"{head}\n\n{fields}{fs}{extra}\n\nReturn exactly one JSON object with keys: {OUT}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


JUDGE_SYS = "You are a strict native-speaker linguist and editor for {name} ({dialect}) with deep knowledge of its code-switching with {switch}. You catch calques, English syntax, invented words, wrong dialect forms, missing diacritics, and fake proverbs. You fix what you can. Output JSON only."

JUDGE_USER = """Evaluate and fix this sample. Dialect target: {dialect}. Switch language: {switch}. Orthography: {orth}

eng: {eng}
m_text: {m_text}
hm: {hm}
ood: {ood}

Score 1-5 (5 = indistinguishable from a skilled native speaker; 3 = understandable but a native would wince; 1 = wrong):
 adequacy (m_text vs eng), m_fluency, hm_naturalness, hm_switch_grammar (are switch points grammatical, matrix language {name}, not word-by-word English syntax),
 hm_idiom, ood_richness (idioms, proverbs, ideophones, vivid native syntax), ood_purity (no raw foreign words), ood_faithful (same situation and intent as eng),
 dialect_consistency, cultural_accuracy (no invented customs/proverbs), diacritics_orthography,
 switch_pin (5 = hm mixes in ONLY the pinned switch language; 1 = another language is mixed in too), eng_plain (is eng PLAIN, official English as a person from England would say it: no idiom, metaphor, slang or flourish? 5 = perfectly plain), no_insult (5 = nothing demeaning or abusive in any field).
 hm must be a natural code-switched variation of m_text that mixes ONLY the pinned switch language; ood must be a fuller informal, idiom-rich native re-expression with the same situation and intent.
m_text must be STANDARD {name}; only hm and ood are pinned to the dialect above, and only where that dialect genuinely differs. Be conservative: change a form ONLY when you are sure it is wrong (invented/misused word, wrong diacritics, calque, English syntax, broken switch point, wrong tense/person/meaning); never swap one acceptable form for another, never impose a dialect on a standard-correct form, never change meaning. List concrete errors with a fix, then give corrected versions of m_text, hm and ood (copy unchanged if already good). Do not shorten ood to be safe; keep its richness.
keep = true only if after correction every field would pass a native reviewer. certain = your confidence 0-1 that YOUR corrections and scores are right (be honest: lower for languages/forms you are unsure of).

Return JSON: {{"scores": {{"adequacy":0,"m_fluency":0,"hm_naturalness":0,"hm_switch_grammar":0,"hm_idiom":0,"ood_richness":0,"ood_purity":0,"ood_faithful":0,"dialect_consistency":0,"cultural_accuracy":0,"diacritics_orthography":0,"eng_plain":0,"switch_pin":0,"no_insult":0}}, "errors": [{{"field":"hm","span":"...","type":"calque|english_syntax|invented_word|dialect|diacritics|culture|other","fix":"..."}}], "corrected": {{"m_text":"","hm":"","ood":""}}, "keep": true, "certain": 0.0}}"""


def judge_prompt(p: Profile, row: dict) -> list[dict]:
    return [{"role": "system", "content": JUDGE_SYS.format(name=p.name, dialect=row["scenario"]["dialect"], switch=row["scenario"]["switch"])},
            {"role": "user", "content": JUDGE_USER.format(name=p.name, dialect=row["scenario"]["dialect"], switch=row["scenario"]["switch"],
                                                          orth=p.orthography, eng=row["eng"], m_text=row["m_text"], hm=row["hm"], ood=row["ood"])}]


def parse_json(t: str) -> dict | None:
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


ADJ_SYS = "You are a strict native-speaker editor for {name}. You compare two versions of a text blind and pick the one a skilled native speaker would prefer. Output JSON only."
ADJ_USER = """English meaning: {eng}
Target dialect for hm/ood: {dialect}. Switch language for hm: {switch}. Orthography: {orth}

For each field below, version X and version Y differ. Choose the better one for naturalness, correctness, idiom, correct diacritics, and (for hm) grammatical code-switching with {name} as matrix language; for m_text it must be standard {name}. Do not prefer a version just because it is longer or because it is more dialect-marked. Choose "tie" only if truly equal.

{blocks}

Return JSON: {{"m_text": "X|Y|tie", "hm": "X|Y|tie", "ood": "X|Y|tie", "why": "one line"}} (omit fields not shown)."""


def adjudicate_prompt(p: Profile, row: dict, pairs: dict) -> list[dict]:
    """pairs: field -> (X, Y), already blind-shuffled by the caller."""
    blocks = "\n\n".join(f"[{f}]\nX: {x}\nY: {y}" for f, (x, y) in pairs.items())
    return [{"role": "system", "content": ADJ_SYS.format(name=p.name)},
            {"role": "user", "content": ADJ_USER.format(name=p.name, eng=row["eng"], dialect=row["scenario"]["dialect"],
                                                         switch=row["scenario"]["switch"], orth=p.orthography, blocks=blocks)}]


# ---------------------------------------------------------------------------------------------------------------------
# v3 prompt: one STATIC system message (definitions + house rules + ALL gold examples) followed by a small per-row task
# card. The static part is byte-identical on every call so vLLM prefix caching / provider prompt caching pays for it once.
# ---------------------------------------------------------------------------------------------------------------------
GOLD_PATH = None   # set lazily to exemplars/gold.jsonl


def load_gold(path=None) -> list[dict]:
    """All rows of exemplars/gold.jsonl. Tolerant of hand-edited files: objects may be single-line or pretty-printed."""
    from pathlib import Path
    p = Path(path) if path else DATA_ROOT / "exemplars" / "gold.jsonl"
    txt, dec, i, out = p.read_text(encoding="utf-8"), json.JSONDecoder(), 0, []
    while True:
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            return out
        o, i = dec.raw_decode(txt, i)
        out.append(o)


STATIC_SYSTEM = """You are building a gold-standard dataset of how African languages are really written and spoken, to train language models. You work as a team of expert native speakers, professional translators and linguists. You never invent words, customs or proverbs; if you are unsure of a form you use a safer native construction and lower your confidence. Output is ONE JSON object and nothing else.

Every sample has four text fields. They are produced IN THIS ORDER and each one builds on the previous one:

1. eng - the FORMAL, NEUTRAL English version of the content (the official English variation of what hm and ood say). It is produced in TWO steps, always in this order: first eng_initial = the content as a person would naturally first put it into English (it may contain emotion, drama, idiom, metaphor, slang or local colour); then eng = that same content REFORMALISED. The purpose: eng is the clean, culture-free English from which a model can later produce the official translation (m_text), the code-switched version (hm) and the full native parlance (ood). So eng must carry NO expressive layer at all, while keeping every fact and the speaker's point:
   - strip every emotive or dramatic word (heartbreaking, tragedy, terrible, unbearable, wonderful...) and replace it with plain factual wording ("severe", "significant", "a critical issue that requires attention");
   - replace every idiom, metaphor, simile, proverb, exaggeration and rhetorical flourish with its literal meaning;
   - replace slang and colloquial verbs/nouns with standard neutral ones; no contractions, exclamations or interjections;
   - no cultural nuance or local colour: use the ordinary English word (a local term with no English equivalent, e.g. "ajo", may stay);
   - keep the same sentences' content, order, person and address (it is still one person speaking to another), in complete explicit sentences;
   - never add or drop facts. The result reads like a careful official statement, not a news article and not a story with feeling.
   Both fields are kept and used differently afterwards. m_text is translated from eng ALONE - ignore eng_initial completely when you write m_text, so that nothing from the expressive wording leaks into the official translation. hm and ood are produced from eng (it fixes the facts, order and speech act) and may use eng_initial as a BASELINE REFERENCE FOR CREATIVITY: where it applies, it is the starting point for idiomatic expressions and for creating or injecting cultural nuance (native idiom, imagery, proverb, ideophone, greeting or praise formula that fits the situation) - especially in ood, and lightly in hm - but they must still say what eng says. Write both English fields in full: eng_initial is where you may be as vivid as you like, which leaves all your attention for the careful reformalisation in eng.
2. m_text - the OFFICIAL translation of eng into the target language (so it is as formal and plain as eng). Faithful, grammatical, natural (not calqued word-for-word), standard orthography, standard dialect.
3. hm (human-mixed) - a perfect code-switched variation of m_text, as a native speaker really talks in conversation (WhatsApp, voice note, phone call). The target language is the base (matrix) language; the pinned SWITCH LANGUAGE for this sample supplies words and short phrases (most insertions are 1-4 words) at natural, realistic switch points (like a native speaker) - technical and modern vocabulary, emphasis, discourse markers, tags, quoted speech, fixed phrases - plus an occasional longer phrase or clause (about one per sample, up to roughly 10 words) exactly where a native would switch whole stretches. Roughly 15-40% of the words come from the switch language (about a third on average, and never more than about 40% of the sample), varied from sentence to sentence. Keep the switch language to the minority: the base language must clearly lead. The base language keeps its own grammatical frame: verbs, tense and aspect markers, pronouns, particles, negation and sentence structure stay in the base language; the switch language fills in nouns, verbs, adjectives, modern vocabulary, discourse markers, tags and the occasional clause. Same content as m_text, but spoken: contractions, particles, interjections, native sentence structures, natural emotional exclamations. Only the pinned switch language may be mixed in - no third language (no Pidgin, no Afrikaans, no English unless it is the pinned one).
4. ood - the complete, informal, native parlance of the same situation: how a fluent, expressive native speaker would really say it out loud. Rich in idiom, proverbs, imagery, ideophones/expressive particles, praise or oath formulae and complex native sentence structures; it may reorder, expand or compress, but it keeps the same situation, facts and communicative intent. It may/can keep a few nativised loans or code-switching exactly as natives do, but it is dominantly native and clearly more native-rich than hm.

HOW TO FORMALISE eng - two worked examples (eng_initial -> eng)
Example 1
eng_initial: I cannot believe how much the weather has changed in our village. When we were children, the rains were predictable and the land was always green. Now, the heat is unbearable and the soil is just dry dust. Even the cattle are dying because there is no grass left for them to eat. It is heartbreaking to see the land we loved become a desert. We used to play in the streams, but now those streams have completely disappeared. This is a tragedy that we cannot ignore.
eng: The environmental conditions in our village have changed significantly. During our childhood, precipitation patterns were predictable and vegetation was abundant. Currently, temperatures are high and the soil is severely eroded. Livestock mortality is rising due to the depletion of pastures. This ecological degradation of our native area is severe. Local streams where we played have dried up entirely. This is a critical issue that requires urgent intervention.
Example 2
eng_initial: You think your parents are outdated, but listen to me. A person who forgets where they come from is like a tree without roots; the first wind will knock them over. You can go to Europe, but you must keep your culture. It is the only thing that will keep you steady when life gets difficult.
eng: You think your parents are old-fashioned, but listen to me. Someone who forgets their background will lose stability easily; the first minor problem will defeat them. You can move to Europe, but you must preserve your traditions. It is the only thing that will support you when life becomes difficult.
Note what changed in each: emotive words became factual wording, figures of speech (the dust and desert imagery, a tree without roots, knock them over) became literal statements, slang/idiom became neutral phrases - and nothing was added or lost.

GOLD EXAMPLES - written and corrected by native speakers. Match their quality, the way eng stays plain, how hm departs from m_text, how rich ood is, and their orthography. Never reuse their content, names or proverbs.
{gold}

WORK CAREFULLY: you can take your time. Plan the situation (frame), write eng_initial, formalise it into eng, then produce m_text (from eng alone), hm and ood (from eng, with eng_initial as a baseline reference for creativity: idioms and cultural nuance where applicable) word by word. Prefer words and forms you are certain exist in the language - especially for lower-resource languages - and use the native-corrected examples for the language when they are given. Before you answer, re-read every field once: diacritics complete, no invented words, only the pinned switch language mixed into hm (scan hm word by word: any English word when English is not pinned must be replaced), no formal English in Pidgin, eng completely free of expression. Fix what you find and note it briefly in self_check.

HOUSE RULES (apply to every field)
- Never insult. No slurs, no abusive or demeaning labels for anyone (for example "fool", "idiot", "stupid", "mumu"); teasing must be affectionate and gentle.
- Use the full standard orthography of the language, including every diacritic / special letter the language uses (see ORTHOGRAPHY in the task). Never ASCII stand-ins.
- Use the pinned dialect only where it truly differs from the standard; m_text stays standard.
- Do not translate English idioms literally. Express the idea the way native speakers do. If you do not know a real idiom, use concrete native imagery rather than a made-up "proverb".
- DIACRITICS FOR EVERY LANGUAGE THAT HAS THEM: whenever hm or ood contains words of a language that uses diacritics or special letters (Yoruba, Igbo, Hausa, Ewe, Twi, Fon, Lingala, Wolof, Bambara, Fula ...), those words carry their FULL diacritics - whether that language is the base language or the pinned switch language (for example Yoruba or Hausa words inside Pidgin). Never write them in plain ASCII.
- ANY PIDGIN (Nigerian, Ghanaian, Cameroonian - as base language OR as the pinned switch language): every word of it must be real Pidgin as native speakers actually speak and write it. Speakers essentially never use formal English there. Verbs stay in their base form with Pidgin markers (don, bin, dey, go, wan, fit, na, no); NEVER English past or participle forms (understood, washed, destroyed, went, left, saw, came, made, took, gave, knew, thought, said, felt, became, any -ed or -ing form), NEVER English copulas or auxiliaries (is, are, was, were, am, I'm, it's, has, have, will, would, can't), and NEVER bookish words (seek, unfortunately, however, therefore, peacefully, settle the matter, more important). Say it the Pidgin way ("Your health pass this land matter", "the flood carry the boundary go", "I dey really sorry for the wahala"). English words that Pidgin keeps in everyday use (money, house, phone, bus, stress, correct, hustle) are fine, in Pidgin grammar.
- LANGUAGES INSIDE hm (PIN RULE): hm may contain ONLY the base language and the PINNED switch language of the task card. If the pinned language is not English, hm has NO English words at all, not even common ones (stress, phone, school, office, okay, so, but, because, problem): write the pinned language's own word or phrase, the base language's own word, or a nativised loan that already stands in m_text. Names of people, places and brands are fine. If you do not know the pinned language's word, rewrite the clause in the base language; do not fall back on English. If the pinned language is Pidgin or Sheng, its English-derived everyday words are part of it. If the pinned language is English, use only informal, everyday English words and phrases that natives drop into their speech (app, stress, scam, on time, network), never bookish English, always inside the base language's grammar.
- ENGLISH INSIDE ood: only the few nativised loans natives really use, never English phrases or sentences.
- ood is allowed - and expected - to be code-switched too, exactly as natives do (with the pinned switch language, and nativised loans), as long as it stays dominantly native and richer than hm.
- Never copy the seed text, names, phone numbers or private details.
- LENGTH IS ENFORCED: the card gives the exact number of sentences for eng (inside its band: 1-3, 2-5, 6-12 or 13-20). Write exactly that many in eng_initial and eng - count them before you answer; never fewer, never more. m_text and hm keep the same sentence structure as eng; ood may join or split at most one sentence. Do not pad short pieces and do not cut long ones.
- FINAL RULE - ACCURACY AND REALISM: everything must be as accurate and realistic as possible across ALL FOUR fields: correct meaning, grammar, spelling and diacritics, and wording a real native speaker would genuinely say or write in that situation. Use only words, forms and idioms you are certain are real; never invent a word, a compound or a proverb. If you are not sure of a word, choose a simpler word you are sure of.

Also return: frame (1-2 English lines: who speaks to whom, the point, mood - written BEFORE eng_initial), self_check (at most 3 short notes on what you re-checked or corrected), ood_relation ("equivalent" | "idiomatic_paraphrase" | "proverb_equivalent"), devices (idioms/proverbs/ideophones actually used in ood, native form + English gloss), conf (your honest 0-1 confidence for m_text, hm, ood - lower it whenever you doubt a word, a dialect form or an idiom), flags (subset of ["uncertain_terms","seed_unusable","politically_sensitive"]), and - as the LAST key of the JSON, written only after everything else is finished - difficulty: ONE number from 1 to 10 saying how hard it was for you to write this sample in this language (1 = easy and fully confident; 4 = a few words or forms you were not sure of; 7 = many uncertain words, invented-sounding forms or guesses; 10 = you could not do it reliably). Be honest: this number is used to filter the dataset, and an honest high number is far more useful than a flattering low one.

Return exactly this JSON shape: """ + OUT


def load_aug_gold(path=None) -> dict:
    """exemplars/augmented_gold.jsonl -> {lang: [rows]} containing ONLY rows the native speaker has corrected (their `status` key removed).
    Rows still marked CANDIDATE are ignored, so nothing uncorrected ever reaches a prompt."""
    from pathlib import Path
    p = Path(path) if path else DATA_ROOT / "exemplars" / "augmented_gold.jsonl"
    out: dict = {}
    if not p.exists():
        return out
    txt, dec, i = p.read_text(encoding="utf-8"), json.JSONDecoder(), 0
    while True:
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            return out
        o, i = dec.raw_decode(txt, i)
        if "status" not in o:
            out.setdefault(o["lang"], []).append(o)


AUG = None
VARIANTS = 3          # when a language has more corrected rows than it shows, rows rotate through this many example sets (each set is one cacheable system prefix)


import os as _os
TOTAL_SHOTS = int(_os.environ.get("CS_TOTAL_SHOTS", 7))      # examples per prompt (6-7); the row's own language takes up to OWN_CAP of them (at least 3 when it has them), the rest are borrowed
OWN_CAP = min(5, TOTAL_SHOTS)
RULES_FIRST = _os.environ.get("CS_RULES_FIRST", "0") == "1"      # variant: shared rules BEFORE the language-specific examples, so ~7k tokens of the system message are identical for every language (prefix cache)
VARIANT_TAG = ("r1" if RULES_FIRST else "") + (f"s{TOTAL_SHOTS}" if TOTAL_SHOTS != 7 else "")


def _aug_k(lang: str) -> int:
    """Most native-corrected examples of the row's OWN language that go into the prompt."""
    return OWN_CAP


def _key(r: dict) -> str:
    import hashlib
    return hashlib.sha1((r["lang"] + r["eng"]).encode()).hexdigest()[:10]


def _pool(lang: str) -> list[dict]:
    """Every native-corrected row of a language: augmented_gold.jsonl (status-free) + the universal gold.jsonl. Stable order."""
    global AUG
    if AUG is None:
        AUG = load_aug_gold()
        for g in load_gold():
            if "status" not in g:
                AUG.setdefault(g["lang"], []).append(g)
    seen, out = set(), []
    for r in sorted(AUG.get(lang, []), key=_key):
        if r["eng"] not in seen:
            seen.add(r["eng"]); out.append(r)
    return out


def variant_for(lang: str, row_id: str = "") -> int:
    """Which example set a row uses (0 when the language has no more rows than it shows)."""
    import hashlib
    return int(hashlib.sha1(row_id.encode()).hexdigest()[:6], 16) % VARIANTS if len(_pool(lang)) > _aug_k(lang) else 0


def fewshots_for(lang: str, variant: int = 0) -> tuple[list[dict], list[tuple[str, dict]]]:
    """(own-language rows, borrowed (language, row) pairs) for one language and variant. Deterministic - nothing random, so the set is cacheable
    and reproducible. High-resource languages short of rows borrow one row each from other high-resource languages; Ghanaian and Cameroonian Pidgin
    borrow Nigerian Pidgin rows."""
    import hashlib
    from csw.datagen.profiles import HIGH, PIDGIN_VARIANTS
    k, pool = _aug_k(lang), _pool(lang)
    if len(pool) > k:
        st = (variant * k) % len(pool); own = [pool[(st + i) % len(pool)] for i in range(k)]
    else:
        own = list(pool)
    extra: list[tuple[str, dict]] = []
    need = TOTAL_SHOTS - len(own)                      # fill to TOTAL_SHOTS with certain-quality rows from other languages (one row per borrowed language)
    if need > 0:
        if lang in PIDGIN_VARIANTS:                    # Ghanaian / Cameroonian Pidgin: Nigerian Pidgin first (up to 3 rows), then high-resource languages
            pc = _pool("pcm")
            for r in pc[:3]:
                extra.append(("pcm", r))
        srcs = sorted((l for l in HIGH if l != lang and _pool(l)), key=lambda l: hashlib.sha1(f"{lang}{variant}{l}".encode()).hexdigest())
        for l in srcs:
            if len(extra) >= need: break
            pl_ = _pool(l); extra.append((l, pl_[variant % len(pl_)]))
        extra = extra[:need]
    return own, extra


def _fmt_example(r: dict, language: str | None = None, sw_ok: bool = True) -> str:
    d = {"language": language} if language else {}
    d.update({kk: r[kk] for kk in ("eng", "m_text", "hm", "ood") if kk in r})
    if sw_ok and r.get("switch"):
        d["switch_language"] = r["switch"]
    for kk in ("ood_relation", "devices"):
        if r.get(kk):
            d[kk] = r[kk]
    return json.dumps(d, ensure_ascii=False)


def gold_block_for(lang: str, variant: int = 0) -> str:
    """The language-specific GOLD block of the system message (examples that are relevant to the language being generated)."""
    from csw.datagen.profiles import P
    own, extra = fewshots_for(lang, variant)
    name = P[lang].name if lang in P else lang
    valid_sw = set(P[lang].switch) if lang in P else set()
    out = ""
    if own:
        out += (f"NATIVE-CORRECTED EXAMPLES IN THIS VERY LANGUAGE - {name} ({len(own)} of them; this is how real, correct {name} looks in all four fields: match their orthography and diacritics, vocabulary, switch style and richness):\n"
                + "\n".join(_fmt_example(r, None, r.get("switch") in valid_sw) for r in own) + "\n")
    if extra:
        out += (f"\nNATIVE-CORRECTED EXAMPLES IN OTHER LANGUAGES ({len(extra)}; high-resource languages{' and Nigerian Pidgin' if lang in __import__('profiles').PIDGIN_VARIANTS else ''}; their quality is certain). They show the depth, structure and switching style expected - write ONLY in {name}, never in these languages:\n"
                + "\n".join(_fmt_example(r, P[l].name) for l, r in extra) + "\n")
    return out.strip()


def fewshot_map() -> dict:
    """{lang: {k, rows, variants: [{own: [...], borrowed: [...]}]}} - exactly which corrected rows each language's prompt shows (stored in exemplars/fewshot_map.json)."""
    from csw.datagen.profiles import TARGETS
    ref = lambda r, l: {"lang": l, "key": _key(r), "eng_start": r["eng"][:70]}
    out = {}
    for lang in TARGETS:
        n = 1 if len(_pool(lang)) <= _aug_k(lang) else VARIANTS
        vs = []
        for v in range(n):
            own, extra = fewshots_for(lang, v)
            vs.append({"own": [ref(r, lang) for r in own], "borrowed": [ref(r, l) for l, r in extra]})
        out[lang] = {"k": _aug_k(lang), "own_rows_available": len(_pool(lang)), "variants": vs}
    return out


def _gold_block(gold: list[dict]) -> str:
    """Language-independent fallback (universal gold.jsonl rows)."""
    from csw.datagen.profiles import P
    keys = ("eng", "m_text", "hm", "ood", "ood_relation", "devices")
    out = []
    for g in gold:
        d = {"language": P[g["lang"]].name if g["lang"] in P else g["lang"], "switch_language": g.get("switch"), "genre": g.get("genre")}
        d.update({k: g[k] for k in keys if k in g})
        out.append(json.dumps(d, ensure_ascii=False))
    return "\n".join(out)


def static_system(gold: list[dict], lang: str | None = None, variant: int = 0) -> str:
    """System message. With `lang` the GOLD block is that language's own corrected examples (dynamic, before the rules); without it, the universal rows."""
    body = STATIC_SYSTEM
    if RULES_FIRST:                                    # move the GOLD EXAMPLES section behind the rules (it ends up right before the task card)
        i, j = body.index("GOLD EXAMPLES - written"), body.index("WORK CAREFULLY")
        body = body[:i] + body[j:] + "\n\n" + body[i:j].rstrip() + "\n\nThe task card follows. Answer it with the JSON object only."
    return body.replace("{gold}", gold_block_for(lang, variant) if lang else _gold_block(gold))


def _switch_orth(name: str) -> str:
    """If the pinned switch language is one of ours and has special letters / diacritics, say so on the card."""
    from csw.datagen.profiles import P
    for q in P.values():
        if q.name.split(" (")[0].lower() == name.lower() or q.name.lower() == name.lower():
            return f"  [its words must carry full diacritics: {q.orthography}]"
    return ""


def gen_prompt(p: Profile, c: dict, gold: list[dict]) -> list[dict]:
    L = LENGTHS[c["length"]][1]
    card = ("TASK - SCENARIO FIELDS (use every one of them to judge register, vocabulary and what to switch)\n"
            f"base language (matrix, target language of m_text / hm / ood): {p.name} [{p.code}]\n"
            f"all code-switch languages attested for {p.name}: {', '.join(p.switch)}\n"
            f"PINNED switch language for THIS sample (mix in nothing else): {c['switch']}{_switch_orth(c['switch'])}\n"
            f"pinned dialect / variety (it also tells you where the speaker lives): {c['dialect']}\n"
            f"ORTHOGRAPHY: {p.orthography}\nstreet/vernacular register for ood: {p.street}\n"
            f"genre: {c['genre']} ({GENRES[c['genre']][1]})\ndomain: {c['domain']}\n"
            f"speaker: {c['persona']}; speaking to: {c['audience']}\nregister: {c['register']}; mood: {c['emotion']}; medium: {c['medium']}\n"
            f"length of eng: EXACTLY {c.get('n_sent', 0) or 'the stated number of'} sentences (band: {L}) - eng_initial and eng each have exactly this many; m_text and hm keep the same sentence structure\n")
    card += "\nSEQUENCE: frame -> eng_initial (natural wording) -> eng (formalised, no expression) -> m_text (from eng alone, ignore eng_initial) -> hm and ood (from eng; eng_initial is a baseline reference for idiom and cultural-nuance creativity where applicable) -> self_check.\n"
    if p.extra:
        card += f"language note: {p.extra}\n"
    if c["track"] == "A":
        card += ("\nHOW TO START: invent a fresh, concrete situation for this card and write eng first. Local concepts you may build the situation around "
                 "(use only the ones that fit naturally - skipping an anchor is better than forcing it): "
                 + "; ".join(f"{a['term']} = {a['gloss']}" for a in c["anchors"]) + ". "
                 "In eng express each concept you use in ordinary plain English words (e.g. 'traditional wedding', 'pounded yam'); keep a native term in eng ONLY if English has no word for it. "
                 "The native terms themselves belong in m_text, hm and ood.")
    elif c["track"] == "B":
        card += (f"\nHOW TO START: the seed below is real {p.name} text. Take its topic and voice as INSPIRATION for a NEW {c['genre']} situation (do not translate or copy it), "
                 f"write eng for that new situation, then the other fields.\n"
                 f"NATIVE-SEED RULE: the seed is genuine writing in {p.name}. Mirror how it is written - its spelling and diacritics, the everyday words it uses, its sentence patterns and register - and prefer words that actually occur in it over words you would have to invent; if the seed shows the word for something, use that word. Never copy its sentences, names or private details.\n"
                 f"SEED: \"\"\"{c['seed']}\"\"\"\nIf the seed is gibberish or the wrong language, return flags:[\"seed_unusable\"] and empty strings.")
    elif c["track"] == "D":
        card += (f"\nHOW TO START: the formal English below is ALREADY WRITTEN - it comes from a sample in another language. Do not change it: copy it exactly into eng. "
                 f"Write m_text, hm and ood in {p.name} for exactly this content (same facts, same speech act), using the scenario fields above for dialect, register, mood and who is speaking to whom.\n"
                 f"ENG: {c['given_eng']}")
    else:
        card += (f"\nHOW TO START: adapt the English seed below into a {c['genre']} situation set in {c['dialect']} ({', '.join(p.persona_regions[:2])}) life: "
                 f"replace foreign names, foods and institutions with local ones, keep the narrative skeleton: write that adapted version as eng_initial, then formalise it into eng (no emotion, idiom or flourish).\n"
                 f"ENGLISH SEED: \"\"\"{c['seed']}\"\"\"")
    return [{"role": "system", "content": static_system(gold, p.code, variant_for(p.code, c.get("id", "")))}, {"role": "user", "content": card}]
