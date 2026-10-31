"""Write DATASET_CARD.md (Hugging Face README) with live numbers.   python -m csw.datagen.make_card"""
from csw.paths import DATA_ROOT
from csw.paths import env_value
import json, sys, statistics as st, collections as C
from pathlib import Path
HERE = DATA_ROOT
from csw.datagen.profiles import P, TARGETS, GOLD_N, SUPPORT, HIGH, MID3K, MID2K, LOW, VERY_LOW, BRIEF
an = json.loads((HERE / "final" / "analytics.json").read_text()); vers = json.loads((HERE / "prompt_versions.json").read_text())
tier = lambda l: "high" if l in HIGH else "mid" if l in MID3K + MID2K else "low" if l in LOW else "very low (1-3 sentences)" if l in VERY_LOW else "Pidgin (12k, last)" if l == "pcm" else "?"
rows = "\n".join(f"| {l} | {P[l].name} | {tier(l)} | {', '.join(P[l].switch)} | {TARGETS[l]:,} | {GOLD_N.get(l, 0)} | {an.get(l, {}).get('generated', 0)} |" for l in TARGETS)
ver = "\n".join(f"| {v['version']} | `{k}` | {v.get('note', '')} |" for k, v in vers.items())
NAME = env_value("CSW_DATASET_NAME", "csw-code-switch")
card = f"""---
pretty_name: {NAME}
license: other
private: true
language: [{', '.join(sorted(TARGETS))}]
task_categories: [translation, text-generation]
tags: [code-switching, african-languages, synthetic, human-corrected]
---
# {NAME} (private, work in progress)

For {len(TARGETS)} African languages (and three Pidgins), each row holds one situation written four ways:

| field | what it is |
|---|---|
| `eng_initial` | the situation as a person would first say it in English (may be emotional, idiomatic) - generated rows only. It is a baseline reference for `hm` and `ood` (it feeds the creativity of `ood`) and is ignored for `m_text` |
| `eng` | the **formal, neutral English** version: no emotive words, idioms, metaphors, slang, contractions or cultural colour; same facts. This is the clean source from which the three variants below are produced, so English-only SFT/RL data can later be translated three ways |
| `m_text` | the **official translation** of `eng` into the language |
| `hm` | **human-mixed**: a natural code-switched variation of `m_text` as a native speaker talks in chat / voice notes, mixing in one *pinned* switch language (e.g. English, French, Arabic, Pidgin) at realistic switch points, with the language's full diacritics |
| `ood` | the **complete informal native parlance** of the same situation: idiom, proverbs, ideophones, vivid structure, light code-switching as natives do |

Other fields: `source` (`human_gold` | `generated` | `generated_given_eng`), `prompt_version`, `scenario` (genre, domain, speaker, audience, register, mood, medium, length, dialect, pinned switch language, track), `anchors`, `seed_id`/`seed_src`, `conf` (model's own confidence per field), `difficulty` (see below), `self_check`, `scores`, `quality_tags`.

## Where the rows come from
* **Human gold**: {sum(v['gold'] for v in an.values())} rows written / corrected by native speakers (6 universal examples + per-language examples). Always kept. `genre`, `switch`, `dialect` on gold rows are unverified hints.
* **Generated**: Gemma-4-31B (vLLM) with a static system prompt (rules + all universal gold rows + the formalisation examples) and a per-row task card: scenario fields, 3-5 native-corrected examples of the row's own language, and a seed:
  * **A** - cultural anchors (model-mined native terms + glosses; unverified) the situation is built around;
  * **B** - a real passage in the language (internal pretraining corpus + public datasets) used only as inspiration (+ a "native-seed rule": mirror its spelling and vocabulary);
  * **C** - an English seed (TinyStories chosen by Flesch-Kincaid grade, or English domain questions on money/law/health/tech/...) adapted to a local setting;
  * **D** - English already generated for another language, given and copied into `eng` (used mostly for Pidgin and the small languages).
* Sampling: temperature 0.6, top-p 0.95, top-k 40, repetition penalty 1.05. Rows from earlier prompt versions are **kept and tagged** (`prompt_version`, `superseded`); nothing is silently dropped.

| version | fingerprint | what changed |
|---|---|---|
{ver}

## Languages, targets and gold
Targets rise with the number of native-corrected examples (very low: 150 per example up to 2,000; mid: up to 4,000; high-resource 6,000; Pidgin 12,000, generated last).

| lang | name | tier | switch languages | target rows | gold rows | generated so far |
|---|---|---|---|---|---|---|
{rows}

## Rule-based scores (`scores`)
No model is involved. Every component is in [0,1]; `scores.rule_score` combines them. Vocabulary lists come from the language's own seed text plus its gold rows. Bands were calibrated on the corrected gold rows (their p10-p90), and the gold rows themselves score {st.mean(v['rule_score_gold'] for v in an.values() if v['rule_score_gold'] is not None):.2f} on average, so the scale is "how much like the gold".

| component | how it is computed |
|---|---|
| `structure` | share of {{eng, m_text, hm, ood}} that are non-empty |
| `eng_formal` | `clip(1 - 0.25*emotive_words - 0.20*slang_words - 0.25*figurative_patterns - 0.10*contractions - 0.15*exclamation_marks)` over `eng`; emotive list (heartbreaking, tragedy, terrible, unbearable, ...), slang list (gonna, wow, ...), regexes for similes/idioms |
| `m_text_length` | `band(words(m_text)/words(eng) / expected, 0.7, 1.4, slack 0.8)`; `expected` = median ratio over that language's gold rows (>=2), else 1.0 |
| `m_text_lexicon` | `clip(1 - max(0, OOV - 0.12)/0.35)`; OOV = share of words (>2 letters) never seen in the language's seed + gold vocabulary (needs >=40 documents) - a proxy for invented words |
| `hm_mix` | switch-language share of `hm`; if the pin is English: share of words that also occur in `eng`; otherwise `max(0, OOV(hm) - OOV(m_text))`; `band(share, 0.12, 0.42, 0.20)` (English pin) or `band(share, 0.08, 0.50, 0.25)` - the band stops near 40% on purpose: gold median 36%, and the model tends to overshoot |
| `hm_vs_m_text` | 0 if `hm == m_text`, else `band(word-level similarity(m_text, hm), 0.07, 0.75, 0.3)` (Pidgins 0.5-0.92) |
| `ood_richness` | 0 if `ood == m_text`, else `0.4*band(words(ood)/words(m_text), 1.05, 2.0, 0.8) + 0.3*clip((type_token_ratio - 0.45)/0.25) + 0.3*clip(n_devices/3)` |
| `diacritics` | for languages with required marks: `0.5*min(1, density(hm)/0.03) + 0.5*min(1, density(ood)/0.03)` (marks per letter); else 1 |
| `repetition` | 0 if any n-gram (n<=4) repeats 4 times in a row in m_text/hm/ood, else the smallest distinct-3-gram ratio |
| `pinned_language` | `1 - 0.5` if English is mixed in although another language was pinned `- 0.3` if Pidgin leaked into a non-Pidgin `hm` |
| `pidgin_purity` | Pidgins only: `clip(1 - 0.3*formal_English_words)` (is/are/was, -ed forms, understood, peacefully, ...) |
| `seed_copy` | 0 if any 7-word sequence of the seed text appears in m_text/hm/ood, else 1 |

`rule_score = (0.20 eng_formal + 0.15 m_text_lexicon + 0.15 hm_mix + 0.10 hm_vs_m_text + 0.10 ood_richness + 0.10 diacritics + 0.05 m_text_length + 0.05 pinned_language + 0.05 repetition + 0.05 pidgin_purity) / (sum of the weights of components that could be computed) x [structure == 1] x seed_copy x [no hard gate failed]`.
Hard gates (row scores 0 and is tagged `gate:...`): an empty field, `hm == m_text`, `ood == m_text`, `m_text` still English (not for the Pidgins), a foreign script where it is not allowed, an `ood` that is mostly English (>40%), >=3 formal-English words in Pidgin.
`quality_tags` lists every flag raised (`eng_expressive`, `hm_over_mixed`, `copies_seed`, `gate:*`). `scores.model_difficulty` repeats the model's own `difficulty`.

## `difficulty` (model-reported)
The last key the model writes: one number 1-10 - how hard the sample was to write (1 easy and confident, 4 a few unsure words, 7 many guesses, 10 unreliable). It is a self-report meant for statistical filtering (e.g. drop the top decile per language), not a guarantee; `scores.model_difficulty` and the per-language correlation with `rule_score` are reported in `final/analytics`.

## Limits
Nothing here has been judged for correctness by an LLM judge or by native speakers beyond the gold rows. Generated text is Gemma-4-31B output; low-resource languages (Efik, Ibibio, Urhobo, Edo, Kimbundu, Umbundu, Mooré, Sango, ...) are the weakest and use native-corrected few-shots and one-sentence rows to compensate. Seeds include non-commercial and research-only sources used as inspiration only; keep this repo private.
"""
(HERE / "DATASET_CARD.md").write_text(card, encoding="utf-8"); print("card written", len(card), "chars")
