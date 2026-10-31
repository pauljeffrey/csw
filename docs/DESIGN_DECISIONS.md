# Design decisions

This document records why the corpus, the quality control and the data formats look the way they do. It contains design rationale only; measured results are reported elsewhere.

## 1. Goal and the three problems it targets

For every language where speakers really mix languages, the corpus provides rows with four aligned fields so that a model learns (a) how people really talk, (b) deep idiom and heritage register, and (c) local concepts that English-centric text never contains.

| Field | Meaning | Rules |
|---|---|---|
| `eng` | The final English reference for the situation | Natural English of a bilingual local; untranslatable local terms stay in their own form |
| `m_text` | Official, standard-register translation of `eng` | Strict; natural (not calqued); standard orthography, standard dialect |
| `hm` (human-mixed) | What a native speaker actually types or says when code-switching | The target language is the matrix; another language supplies content words, markers, tags and quotes at grammatical switch points; restructured, not word-swapped; full diacritics |
| `ood` | The richest street-and-heritage rendering a deep native could give | Dense idiom, proverb, ideophone, praise and oath formulae; vivid native syntax; same situation and intent as `eng` but free in expression (not a translation); full diacritics; no raw foreign words (fully nativised loans only) |

Per row: confidence per field, `ood_relation` (equivalent / idiomatic paraphrase / proverb equivalent), the devices used in `ood`, the scenario card, and gate and judge metadata.

Problems addressed:

1. **Lost local culture.** English is not the source. One track builds the English *from* a per-language bank of cultural anchors; another starts from real native text.
2. **Translationese** (English word order, repetitive vocabulary). The native-seeded track is native-first, and `hm` / `ood` are generated from a language-neutral meaning `frame` plus `eng`, with an explicit instruction to restructure and never to transform `m_text`.
3. **Western-only entities.** The English-writing prompt bans Western defaults; personas and regions are locale-specific; anchors carry an `untranslatable` flag.

## 2. Language set and profiles

The set contains the languages in which switching is attested, together with the switch languages that are attested for each (for example English, French, Arabic, Afrikaans, a regional Pidgin or a neighbouring African language). Each language has a profile (`csw.datagen.profiles`): a dialect set (first = standard), a "street" register definition, orthography and diacritic rules, persona regions and a resource tier. Every row pins **one** dialect and **one** switch language; when several switch languages are attested one is drawn per row.

Pidgins are English-lexified: for them `m_text` is standard written Pidgin, `hm` is Pidgin mixed with standard local English and `ood` is deep basilect. Ghanaian and Cameroonian Pidgin are separate languages, and English is not a pinned switch language for Pidgin.

Languages that the generator cannot write (measured by the out-of-vocabulary rate of generated text against real text of that language) are held and weighted down or out rather than silently kept; the three support tiers map to row weights 1.0 / 0.5 / 0.

## 3. Where the English and the seed text come from

Three tracks mix per language (a track falls back to the anchor track when a language has no seeds of that kind), plus a fourth for reuse:

* **A: anchor-seeded English.** Per language, a bank of cultural anchors (term in native orthography, English gloss, category, `untranslatable` flag) mined from about 20 life domains (food, kinship, ceremonies, informal finance, transport, customary law, folklore, flora and fauna, music, health, farming, school and work, politics, youth technology, greetings and insults, proverbs, festivals, home, ...). Each row draws one to three anchors and a **scenario card**; the generator writes the English first (as a bilingual local, anchors kept untranslated), then the other fields.
* **B: native-seeded.** A real native-language passage (conversation, proverb, story, comment) is inspiration; the generator invents a *new* situation in its voice and writes `eng` as a by-product. This is the only track with zero English syntax in the source. Seeds are inspiration only and never copied.
* **C: English text localisation.** English story or domain text is *localised*: Western names, foods, institutions and settings are replaced while the narrative skeleton is kept; `eng` is the localised English.
* **D: reused English.** English already produced for related languages is given as the English for Pidgins and small languages.

Seed licensing is explicit: sources are tagged commercial-OK or non-commercial, and non-commercial sources can be switched on or off by a flag. Seeds from noisy corpora pass language-identification and stop-word filters before use. Song lyrics are treated as *pattern mining only* (switch-point statistics, idiom inventories), never as training text.

## 4. Scenario grid

A scenario card combines genre (chat, explain/teach, story, advice/family, opinion, emotion/ceremony, market, instructions, tradition/law, humour, faith), length (short, medium, long; low-resource tiers use short rows only), a domain from a cross-cutting list, a persona (age, gender, occupation), register, mood and medium (chat message, voice note, phone call, radio phone-in, ...). The plan is stratified with exact quotas, deterministic per `(lang, seed)`, ids are content hashes, a seed text is reused at most a few times before the row falls back to anchors, and every stage resumes.

## 5. Generation: one call, all fields

The system message plays three native speakers in one (a translator, an urban young adult who mixes languages, an elder or street raconteur) and forbids invented words or fake proverbs. The output order forces planning: `frame`, then `eng`, `m_text`, `hm`, `ood`, then confidences, `ood_relation`, `devices` and flags. `ood` and `hm` are specified from the meaning, not as edits of `m_text`. The teacher model is configurable. Long rows need a large output budget because African-language text tokenises into many tokens.

The prompt is versioned by fingerprint; every row records the fingerprint of the prompt that made it, and rows are tagged with the prompt version rather than being deleted, so a downstream step can filter by version or regenerate only one field.

The pinned switch language is enforced twice: in the prompt and by a deterministic stop-word gate, because a judge model does not reliably notice English leaking into a row that was meant to switch into another language.

## 6. Quality control without human raters

Four layers, cheapest first; each leaves a record that can later supervise a quality estimator.

1. **Deterministic gates** (`gates`): empty fields; `hm == m_text` or `ood == m_text`; `m_text` still English; wrong script; English function-word leak in `ood` / `m_text`; `hm` unmixed or over-mixed; repetition; length ratio; missing diacritic density; ASCII dot-under hacks; NFC normalisation. Hard failures are dropped, warnings reduce the score.
2. **Frontier judge from a different model family than the generator.** Rubric scores per field on a 1 to 5 scale, a typed error list, a conservative corrected version of the field, a keep decision and a self-reported certainty (used only as a discount).
3. **Blind adjudication by a third model.** For every field the judge edited, a randomised X/Y blind vote between the original and the correction. Only wins are accepted; losers are kept as rejected. This guards against the main failure of a model used as editor: confident "fixes" that are wrong (for example imposing a regional dialect form on a standard-register field or changing tense).
4. **Final score**: a weighted mean of the judge scores (discounted by certainty and by gate warnings) and the generator's lowest confidence; a row is kept if it has no hard gate failure and the judge keeps it. The score threshold is a placeholder; a per-language percentile is preferred once distributions are known.

Every row therefore carries original, corrected, winner, scores and gate metrics, from which preference pairs (`chosen` / `rejected` per field) are derived.

Judges are used through **batch APIs only** (half price, 24 hour window); real-time calls are not used for judging. Large pools are sharded to respect per-model active-token quotas.

Judges are calibrated, not trusted: graded corruption ladders measure whether a judge's grade moves with the damage, a back-translation check and a corrupted-copy detection test measure what each judge catches, and capability and repeatability screens decide which judge may label which language (rows in a language the judge cannot read are not used as labels).

## 7. The evaluator

A single multilingual encoder scores 27 dimensions on the judge scale from `[task | language | pin  source  target  devices]`. Design choices:

* **Group tokens** summarise each field's dimensions, and only the dimensions of the target field are scored and trained (masked multi-task regression).
* **Devices** (the plan phrases or idioms the generator committed to) are dropped from a share of training examples so the same model scores text whose plan is unknown; the device-related dimensions are kept as labels in that case.
* **Low-end supervision** comes from synthetic corruption chains with known order and from judge-labelled natural bad translations, so the evaluator sees the whole grade range and not only "fine" and "destroyed".
* **Human-rated external data** is used for translation only and is kept out of anything not cleared for production use; sources whose licences forbid production use are excluded.
* **Self-training** is evaluated under a pre-registered protocol: an ensemble of labelled-only teachers pseudo-labels the unlabelled pool, confidence is the agreement of the model with itself over views of the same input, the threshold is tuned on dev, and the self-trained model is adopted only if it beats the base with a bootstrap confidence interval that excludes zero.

## 8. Data formats

Plan-first completions (language tags, then the plan, then the text) make three consistency checks possible: planned items must occur in the written text, switch-language words in the text should have been announced, and, for `hm` with an English pin, planned items must come from the English source. The task tags and plan brackets are real single tokens (added as normal, non-special tokens so that decoding with special tokens skipped does not delete them). The translation task keeps the exact format of the base translation checkpoint so its behaviour is preserved, and a translation replay from human-translated data protects it during SFT. Splits are by hash of the English text so one English sentence never appears in both train and eval, and train is additionally isolated from test by word 8-gram overlap.

## 9. Reinforcement learning

The reward is the evaluator score (equal-weight mean of the task's dimensions) minus deterministic rule penalties whose total is capped so the rules never outweigh the evaluator, with a floor for any format failure. Rules use per-language tables (function words, diacritic blocks, number words) and report "cannot be computed" instead of guessing when a table is missing. Groups contain sampled completions plus the greedy completion, which joins the baseline only through the sampled statistics. A KL guard skips the optimiser step when the policy moves too far from the reference, and a fixed monitor set tracks evaluator score against KL to expose over-optimisation. RL uses fresh English that was not used to generate the SFT rows.

## 10. Data hygiene

Datasets are never stored in this repository. Test material of public benchmarks is removed on both sides when building replay data, sources are allow-listed by licence, and personal data is scrubbed from seeds (URLs, phone numbers, handles).
