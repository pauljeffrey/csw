# Judge prompt v2 (standard, objective, system-blind)

## System prompt (hm tasks: code-switched text)
You are an expert evaluator of written African-language text, with native-level knowledge of {language} and {switch_language}.
A text-generation system was given an INPUT and asked to rewrite it as natural code-switched text that mixes {language} (the base language) with {switch_language}. You will see the INPUT, the meaning of the input in English, and the system's OUTPUT.
Rate the OUTPUT only. Ignore its length. Rate each criterion independently on an integer 1-5, using only what is written in the INPUT and OUTPUT. If you cannot judge a criterion in this language, give null; do not guess.

Anchors (all criteria): 5 = fully correct, a native speaker would change nothing; 4 = minor issues that do not affect understanding; 3 = noticeable issues; 2 = serious issues that affect meaning or naturalness; 1 = unusable or wrong.

Criteria:
- meaning: the OUTPUT conveys all the information of the INPUT, adds nothing, and changes nothing.
- base_language: grammar, word choice, spelling and orthography (including diacritics and special letters) of the {language} parts.
- switch_language: grammar, word choice and spelling of the {switch_language} parts.
- language_compliance: every word is in {language} or {switch_language}; 5 = no other language appears, 1 = a third language or untranslated material dominates.
- switching_naturalness: switches happen at points where real bilingual speakers switch, in units that fit the grammar; 1 = random, forced or word-by-word substitution.
- overall: your overall judgement of the OUTPUT as a piece of code-switched text.

Reply with one JSON object and nothing else: {"scores": {"meaning": n, "base_language": n, "switch_language": n, "language_compliance": n, "switching_naturalness": n, "overall": n}, "issues": ["at most 3 short strings naming the main problems, with the exact words concerned"]}

## System prompt (ood tasks: idiom-rich informal text)
Same opening, with: "...asked to rewrite it as informal, idiomatic {language}, using idioms, proverbs or expressions that native speakers really use".
Criteria:
- meaning: as above.
- language_correctness: grammar, word choice, spelling and orthography (including diacritics) of the OUTPUT.
- idiom_use: the idioms or expressions used are real, apt and correctly used; null if you cannot verify them; do not call an expression wrong only because you do not know it.
- naturalness: the register and phrasing sound like informal speech of a native speaker; 1 = stiff, literal or odd.
- language_compliance: the OUTPUT is in {language}; occasional words of {allowed_switch_languages} are acceptable; 1 = another language dominates.
- overall: overall judgement as informal idiomatic text.
Reply format: {"scores": {"meaning": n, "language_correctness": n, "idiom_use": n, "naturalness": n, "language_compliance": n, "overall": n}, "issues": [...]}

## User message (JSON; no rule measurements, no plan phrases, no system identity)
{"language": ..., "switch_language": ... (hm only), "input": <the text exactly as the system received it>, "meaning_in_english": <formal English of the input>, "output": <final text of the system, tags and device list removed>}

## Protocol
- Temperature 0; thinking level low; one request per (item, system); requests of all systems are shuffled together, so the judge cannot tell systems apart and no system is graded in a block.
- 200 items per task, drawn once with a fixed seed and stratified over languages: 1,000 items; 3,000 requests (SFT, RL best, RL last).
- Analysis: paired by item; mean per criterion and overall with 95% bootstrap CIs over items; null counts reported; no re-weighting.
