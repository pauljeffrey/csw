# Architecture

This document describes the stages of the project, how data moves between them, which module does what, and the string formats that tie the stages together. It describes structure only; datasets, outputs and results are not part of this repository.

## 1. Overview

The project builds a corpus for code-switching and idiom generation in African languages and trains a generator on it. Each corpus row describes one situation in four aligned fields:

| field | meaning |
|---|---|
| `eng` | the English reference for the situation |
| `m_text` | a standard-register, non-calqued translation of `eng` (the "matrix" rendering) |
| `hm` | a natural code-switched rendering (matrix language plus one pinned switch language) |
| `ood` | a dense idiomatic / heritage-register rendering that is not a translation of `eng` |

plus per-row metadata (scenario card, pinned switch language, dialect, devices, confidences, gate metrics).

The pipeline has five stages. Every stage reads and writes plain files below one data root (see section 5); no stage keeps state anywhere else.

```
 seeds, anchors ─┐
 pretraining     │   1. DATA GENERATION            rows (*.raw.jsonl)
 corpus slices ──┴─► scenario plan ─► vLLM / API ─► deterministic gates ─► tagged rows
                                                     │
                                                     ▼
                              2. JUDGING   rubric judges through batch APIs (Gemini, GPT),
                                           corruption ladders for calibration, span edits
                                                     │ judged rows, per-dimension grades
                          ┌──────────────────────────┼──────────────────────────┐
                          ▼                          ▼                          ▼
                3. SFT data + training      4. EVALUATOR training      (judge-corrected rows,
                (5 tasks + translation      (27 graded dimensions,      preference pairs)
                 replay)                     supervised + self-training)
                          │                          │
                          └──────────► 5. RL (GRPO) ◄┘
                                      reward = evaluator score - capped rule penalties
```

## 2. Stages

### 2.1 Data generation (`csw.datagen`)

* **Language profiles** (`profiles`): per language the dialect set (first = standard), orthography and diacritic rules, switch languages, persona regions, resource tier, per-language row targets and generator-support tier.
* **Seeds and anchors** (`seeds`, `build_seeds`, `extract_slices`, `build_en_domain`, `tiny_fk`, `culture_seeds`, `build_culture_pool`): native-language passages streamed from Hub datasets or sliced from the tokenised pretraining corpus, English domain/story seeds, a per-language bank of cultural anchors (terms, glosses, categories) and culture seed cards.
* **Scenario plan** (`cs_pipeline`): a deterministic stratified plan per `(language, seed)` over genre, length, domain, persona, dialect, register, mood and medium. Ids are content hashes and every stage is resumable (JSONL keyed by id; ids already present are skipped).
* **Generation** (`cs_prompts`, `cs_pipeline`, `providers`): a static system message (definitions, house rules, gold few-shot rows) plus a per-row task card. The generator is served with vLLM through an OpenAI-compatible endpoint (`provider local`), or reached through the OpenRouter / Together / Doubleword providers. One call returns all four fields plus `frame`, confidences, `ood_relation`, `devices` and flags. Sampling parameters are chosen empirically on a pilot (`pilot_sampling`, `pilot_select`).
* **Deterministic gates** (`gates`, `tag_rows`, `score_rows` in the judging package): empty fields, `hm == m_text`, English leakage when English is not the pinned switch language, wrong script, over- or under-mixing, repetition, length ratios, diacritic density, NFC. Rows are tagged rather than deleted; downstream steps filter on tags.
* **Pools and exports** (`build_eng_pool`, `build_augmented_gold`, `build_highres_candidates`, `export_prompts`, `assemble_final`, `make_card`): English pools reused for related languages, few-shot candidate banks, the exact per-row prompts, and the assembled per-language dataset.
* **CPU monolingual writer** (`cpu_gen`, `budgets`): small-model monolingual text for pretraining, served with llama.cpp; per-language token budgets.

| module | role |
|---|---|
| `ab_test` | A/B test of prompt variants on cards that were ALREADY generated (same ids, same cards). |
| `assemble_final` | Assemble the final per-language dataset from the judged and adjudicated rows. |
| `bible_test` | Can Gemma translate Bible-style English into Efik / Urhobo?  Zero-shot vs 5-shot with retrieved REAL parallel verses;. |
| `bt_sets` | Build raw files for the back-translation check (gold rows versus generated rows). |
| `budgets` | Per-language generation budget for the CPU writer. |
| `build_audit_review` | Merge every Gemini audit result into ONE reviewable file (no API calls). |
| `build_augmented_gold` | Pick the cleanest generated rows per language as candidate few-shots for the native speaker to correct. |
| `build_culture_pool` | 80k culture seed cards for the CPU writer - built offline (no GPU, no model) |
| `build_en_domain` | English DOMAIN seed pool (money / law / health / tech / family / politics / school / sport / work / stories / chat), streamed from labelled. |
| `build_eng_pool` | Build the pool of English texts reused as given English for the Pidgin and low-resource languages. |
| `build_highres_candidates` | Candidate few-shot rows for the HIGH-resource languages, for the native speaker to correct. |
| `build_seeds` | Sample slices, decode, split into documents, filter, and write seed JSONL per language. |
| `clean_augmented_gold` | Tolerant loader + cleaner for the hand-edited exemplars/augmented_gold.jsonl. |
| `cpu_gen` | Monolingual African-culture text for pretraining, written by a small Gemma model on the box's idle CPU (llama.cpp server). |
| `cs_pipeline` | Pipeline driver: anchors, seeds, plan, generate, judge, adjudicate, report. |
| `cs_prompts` | Prompts for anchor mining, generation and judging. |
| `culture_seeds` | Diverse culture seed cards written by the large generator. |
| `exemplars` | Build exemplar banks (few-shot rows) from adjudicated rows. |
| `export_prompts` | Export the exact per-sample prompt (the user message / task card) for every planned row. |
| `extract_slices` | Sample token slices per language and source bucket from tokenised pretraining files. |
| `gates` | Deterministic gates run before (and after) the LLM judge. |
| `lang_support` | Which languages does the generator actually handle? |
| `make_card` | Write DATASET_CARD.md (Hugging Face README) with live numbers. |
| `make_preview_md` | Markdown preview of raw rows with gate flags and generator-support tier. |
| `merge_rehm` | Keep the old rows (tagged when their hm is wrong) and add NEW rows with the regenerated hm. |
| `mock_server` | Tiny OpenAI-compatible mock (no GPU) |
| `pilot_sampling` | Pick the sampling configuration empirically on the real server. |
| `pilot_select` | The 60 pilot task cards (12 languages x 5), shared by pilot_sampling.py and export. |
| `profiles` | Per-language profiles for the code-switching corpus (eng / m_text / hm / ood). |
| `render_preview` | Regenerate prompt_preview.md from the LIVE prompt builders + exemplars/gold.jsonl + prompt_changelog.md. |
| `seed_stats` | Seed statistics per language. |
| `seeds` | Seed pools (streamed from Hub datasets and built from the pretraining corpus). |
| `tag_rows` | Tag every generated row with prompt version and quality tags instead of deleting rows. |
| `tiny_fk` | Simple English seeds ranked by Flesch-Kincaid grade. |
| `verify_all` | Pre-flight checks before spending GPU time. |
| `view_raw` | Readable markdown from raw generated rows (no judge needed). |
| `view_samples` | Render judged rows into a readable markdown file + print structure statistics. |

### 2.2 LLM-judge scoring (`csw.judging`)

* **Rubric judge**: a judge card (prompt plus JSON schema, v1.3) scores the rubric dimensions (M1-M6 for `m_text`, H1-H9 for `hm`, O1-O7 for `ood`, D1/D2 for devices, X1 for cross-field consistency), lists typed errors with spans, and may propose span edits. `apply_edits` applies those edits deterministically and all-or-nothing.
* **Two judge families through batch APIs only**: Gemini (native Batch API in `gemini_batch`, or a `:batch` variant through OpenRouter in `run_judge_openrouter`) and OpenAI GPT models (`oai_judge`, `batch_judge`). Large pools are sharded to respect per-model active-token quotas (`batch_shards`, `batch_driver`). Results are written as `results_<judge>@v13.jsonl` and read back by `analyze_judge`, `probe_report` and the evaluator data builders.
* **Calibration**: `corrupt_ladder` creates graded synthetic corruptions (three intensities per rubric dimension, language-agnostic operators) so judge sensitivity and the evaluator's low end can be measured; `oai_corrupt` and `oai_bt` run detection and back-translation checks; `probe_report` screens capability and repeatability.
* **Selection**: `select_judge_pool` and `select_round2` choose which rows are judged (stratified by proxy tier and gates); `build_compact_requests` / `compact_judge` build cheap single-task prompts.
* **Gold audit**: `repair_gold`, `audit_score`, `audit_summary`, `audit_worker` audit and correct the few-shot gold rows.

| module | role |
|---|---|
| `ab_judge` | Judge the A/B test rows. |
| `analyze_judge` | Read judge results and report. |
| `audit_score` | Scoring pass for the batch-audit proposals. |
| `audit_summary` | Per-language summary of the gold audit. |
| `audit_worker` | Poll the gold-audit batches and write results when each finishes. |
| `batch_driver` | Submit shard directories within the token quota, poll, and download finished shards. |
| `batch_judge` | Judge a requests file through the batch APIs only. |
| `batch_shards` | Submit a large judge pool as shards that respect the provider's active-token quota. |
| `build_compact_requests` | Batch request shards with the compact prompts. |
| `build_judge_batches` | Batch-API request files for scoring rows with two judges, plus a calibration set with known answers. |
| `build_pool_requests` | Turn the judge pool and rows into judge requests. |
| `collect_m_raw` | Parse M1..M6 batch results into labels. |
| `compact_judge` | Compact per-task judge prompts. |
| `corrupt_ladder` | Graded synthetic corruptions per rubric dimension. |
| `fetch_english` | Fresh English source texts for the RL prompts, taken from licensed triage datasets. |
| `gemini_batch` | Native Gemini Batch API for the judge stages. |
| `judge_pilot2` | Second pilot of the judge card. |
| `judge_v11_test` | Small live test of the judge card against the Gemini API. |
| `label_replay` | Label human replay pairs with a short prompt through the batch API. |
| `oai_bt` | Back-translate then compare, through the OpenAI batch API. |
| `oai_corrupt` | Does the OpenAI judge catch real errors?  Builds corrupted copies of gold rows and counts detections. |
| `oai_judge` | OpenAI Batch API as a labeller (build, submit, poll, report). |
| `oai_worker` | Poll the OpenAI judge batch and write labels when it completes. |
| `probe_report` | Capability screen and repeatability for the judges. |
| `repair_gold` | Score, judge and correct gold rows with a strong model. |
| `run_judge_openrouter` | Run a judge-batch stage through OpenRouter. |
| `score_rows` | Rule-based quality scores for rows (no model involved). |
| `select_judge_pool` | Select the rows that the judges will score. |
| `select_round2` | Select rows for the second judging round. |

### 2.3 Supervised fine-tuning (`csw.sft`)

* **Formats** (`fmt`, `fmt2`, `tokens`, `devices`): prompt and completion strings (section 4), the added tokens (`<hm>`, `<ood>`, `<dev>`, `</dev>`, language tags, `<draft>`), and extraction of the plan ("devices") from generated rows so the model can learn to plan before it writes.
* **SFT data** (`build_sft`, `build_sft2`, `judge_fixes`, `build_edits_sft`, `build_ablation`, `build_replay*`, `add_replay`, `merge_repairs`): per-task examples from kept rows, judge-corrected copies and rejected fields, split by hash of the English text so one English sentence never lands in both train and eval, translation replay from human-translated data to keep the base translation behaviour, and leak audits (`audit_split`, `isolate_train`).
* **Training** (`make_base`, `upscale`, `train_sft`): the starting checkpoint with renamed tokenizer slots, optional depth up-scaling, and a single-GPU token-budget trainer with loss on the completion only.
* **Evaluation and generation** (`eval_chrf`, `compare_arms`, `gen_test`, `gen_vllm`, `mt_gen`, `africomet_score`): cheap chrF-based evaluation, ablation-arm comparison, held-out generation (HF and vLLM), and reference-free AfriCOMET scores.

| module | role |
|---|---|
| `add_replay` | Add the enlarged replay to an SFT build without touching validation and test. |
| `africomet_score` | Reference-free AfriCOMET scores for (english, translation) pairs. |
| `apply_edits` | Deterministic, all-or-nothing applier for the judge's span edits (rubric_v1.md section 5.3). |
| `audit_split` | Leak audit of an SFT build. |
| `build_ablation` | Three SFT arms on the same stratified subsample, differing only in the format of the ood tasks. |
| `build_edits_sft` | csw SFT examples from the strict-Flash corrected rows. |
| `build_nb_pool` | Pool of unused cleared pairs for the evaluator's low end. |
| `build_replay` | Candidate translation replay pairs from human-translated data. |
| `build_replay2` | Replay candidates from the decontaminated English-to-X set. |
| `build_replay3` | Enlarge the translation replay from the cleared candidate pool. |
| `build_replay4` | Uniform per-language enlargement of the translation replay. |
| `build_rl_prompts` | RL (GRPO) prompt sets for the code-switcher. |
| `build_sft` | Build the SFT set for the code-switcher (eng -> m_text | hm | ood) from the generated rows (+ gold). |
| `build_sft2` | SFT set builder for the five-task format. |
| `compare_arms` | Compare ablation arms on the rows they all generated. |
| `devices` | Extract the PLAN ("devices") for each task from the generated rows, so the model can learn to plan before it writes. |
| `eval_chrf` | Cheap evaluation of csw generations (no model, no API) |
| `fmt` | Prompt / completion formats for the code-switcher (eng -> m_text | hm | ood) on top of gemma-3-270m-translate-rl2b. |
| `fmt2` | Prompt / completion formats of the code-switcher csw (gemma-3-270m-translate-rl2c + 5 new tasks). |
| `gen_test` | Generate for the held-out test prompts and keep everything (for the website of human preferences and for QE scoring). |
| `gen_vllm` | vLLM version of gen_test. |
| `isolate_train` | Remove from train every example sharing a word 8-gram with a test example. |
| `judge_fixes` | What the two LLM judges said about a row, turned into (a) corrected copies of the row, (b) the fields they rejected, (c) a curated evaluation set. |
| `make_base` | Build the starting checkpoint (tokenizer slots renamed, embeddings initialised). |
| `merge_repairs` | Merge the hm repair into the rows without discarding the old values. |
| `mt_gen` | Translations of English sentences by the translator model (BeardedMonster/gemma-3-270m-translate-rl2c) for the evaluator's natural-low-quality data. |
| `push` | Push a trained checkpoint to the Hub. |
| `rewards` | Rule-based rewards for GRPO on the code-switcher (used next to the learned quality estimator). |
| `tokens` | Added tokens of the code-switcher (task tags, plan brackets, language tags). |
| `train_sft` | SFT of csw on its five tasks (+ eng->m replay). |
| `upscale` | Depth up-scaling of a checkpoint. |

### 2.4 Evaluator (`csw.evaluator`)

* **Model** (`qe_lib`): one multilingual encoder with group tokens, 27 graded dimensions on the judge scale (1-5) and a masked multi-task regression loss; only the dimensions that belong to the target field are scored.
* **Data** (`build_qe_data`, `build_human`, `build_chains`, `build_replay_records`, `build_cft`, `build_cft2`, `build_unlabelled`, `collect_*`): judged rows, human-rated translation rows from external benchmarks, corruption chains (known order original > mild > moderate > severe), judge-labelled replay pairs and an unlabelled pool.
* **Training** (`train_qe`, `selftrain`, `ssl_label`, `ssl_select`, `ssl_run`): supervised training, plus a self-training study in which an ensemble of teachers pseudo-labels the unlabelled pool and only confident units are kept.
* **Evaluation** (`eval_qe`, `eval_chains`, `eval_lowend`, `eval_pilot`, `eval_val_slices`, `compare_two`, `ssl_eval`, `ssa_analysis`, `ssa_full_analysis`, `comet_score_all`, `internals`): agreement with the judges, ordering of held-out corruption chains, the low end of the grade range, paired comparisons with bootstrap confidence intervals, comparison with human ratings and external metrics, and model-internals analysis.

| module | role |
|---|---|
| `build_cft` | Data for the first continued fine-tuning stage of the evaluator. |
| `build_cft2` | Data for the second continued fine-tuning stage of the evaluator. |
| `build_chains` | Corruption chains for the evaluator's low end. |
| `build_human` | Human-rated evaluation rows as evaluator training and dev data (translation only). |
| `build_qe_data` | Judged rows to evaluator training data. |
| `build_replay_records` | Labelled replay pairs to evaluator training records. |
| `build_unlabelled` | Unlabelled pool for the semi-supervised evaluator. |
| `collect_compact` | Collect compact-prompt chain batches into evaluator records. |
| `collect_stage2_both` | Parse strict compact results into evaluator records. |
| `comet_score_all` | Score segments with SSA-COMET / AfriCOMET checkpoints. |
| `compare_two` | Paired per-task comparison of two evaluator runs. |
| `eval_chains` | Do the evaluators order held-out corruption chains correctly. |
| `eval_lowend` | Low-end check on held-out natural-bad translation pairs. |
| `eval_pilot` | Evaluators versus the judges on the pilot rows. |
| `eval_qe` | Evaluate a trained evaluator against the judges on the held-out test set. |
| `eval_val_slices` | Mean per-dimension Spearman of runs on slices of a validation file. |
| `internals` | Internals of the final evaluator: what each group token attends to. |
| `push_dataset` | Stage and push training, label and test files to a private dataset repo. |
| `qe_lib` | Evaluator model, input encoding, 27 graded dimensions and masked multi-task loss. |
| `selftrain` | Self-training of one evaluator with confidence-filtered pseudo-labels. |
| `ssa_analysis` | Evaluator versus human ratings and external metrics on an external benchmark subset. |
| `ssa_full_analysis` | Full external-benchmark analysis. |
| `ssl_eval` | Single-use test evaluation of evaluator runs with paired bootstrap. |
| `ssl_label` | Pseudo-labels from an ensemble of teachers, with their disagreement as uncertainty. |
| `ssl_run` | Orchestrator for the semi-supervised evaluator study. |
| `ssl_select` | Select pseudo-labelled units by confidence. |
| `train_qe` | Train the evaluator (supervised, optionally with pseudo-labelled units). |

### 2.5 Reinforcement learning (`csw.rl`)

* **Algorithm**: GRPO with group-relative advantages, a greedy completion appended to every group, a KL penalty and a KL guard that skips an optimiser step when the mean per-token KL exceeds a threshold, and a fixed monitor set decoded greedily at intervals (evaluator score against KL). The generic GRPO, sampling and vLLM components are vendored (`csw.vendor.gemma_mt`, see `VENDORED.md`).
* **Reward** (`reward`, `rules`, `diacritics`, `lid`, `config`, `evaluator`, `qe_client`, `qe_worker`): the evaluator score (equal-weight mean of the task's dimensions, scaled to 0..1) minus deterministic rule penalties that are capped so rules never outweigh the evaluator; any format failure maps to a floor value. Rules cover format (exactly one `<dev>` block with at least two devices, non-empty final text, end-of-turn), draft length, language identification (GlotLID), diacritics, numbers, copying from the source and length windows. Translation (`eng_m`) additionally uses a reference-free AfriCOMET score computed in a separate process because it needs its own environment.
* **Prompts** (`build_prompts`, `build_fresh`, `build_windows`, `mine_numbers`, `calibrate`, `calibrate2`): language-balanced prompt pools with fresh English, length windows and number-word tables derived from SFT references, and calibration of the rule penalties on reference versus generated text.
* **Small tables** (`rl/data/*.json`): function words, allowed switch words, diacritic blocks, number words and base function-word floors; these are configuration shipped with the code.

| module | role |
|---|---|
| `build_fresh` | Build the fresh RL prompt pool. |
| `build_prompts` | RL prompt pool. |
| `build_windows` | Per-language length windows derived from SFT references. |
| `calibrate` | Penalty distributions of the deterministic rules on real full_E generations (and on the references), plus data-derived length windows. |
| `calibrate2` | Rate and mean of every rule penalty on references versus generations. |
| `config` | Reward configuration. |
| `diacritics` | Diacritic penalty. |
| `evaluator` | The trained group-token evaluator as a reward scorer. |
| `lid` | Language-identification guard (GlotLID, fastText). |
| `mine_numbers` | Number words per language, mined from translation pairs by association. |
| `qe_client` | Client of the COMET scoring worker (separate process). |
| `qe_worker` | AfriCOMET-QE scoring worker run in its own environment. |
| `reward` | Reward = evaluator score (equal-weight mean of the task's dimensions, scaled to 0..1) minus capped deterministic rule penalties;. |
| `rules` | Deterministic reward rules for the csw RL. |
| `train_rl` | GRPO for the csw code-switcher. |
| `view_gens` | Readable view of generations logged during RL. |

## 3. Data flow between stages

| from | file / record | to |
|---|---|---|
| data generation | `<out>/<lang>.raw.jsonl` rows (`eng, m_text, hm, ood, scenario, conf, devices, flags`) | gates, tagging, judging, SFT |
| gates / tagging | `<lang>.tagged.jsonl` with `prompt_version`, `quality_tags`, `gate_metrics` | SFT filters, judge pool selection |
| judging | `requests.jsonl` per stage, `results_<judge>@v13.jsonl` | evaluator data, SFT corrections (`judge_fixes`) |
| judging | corruption chains with known order | evaluator low end, judge calibration |
| SFT | `train/val/test.jsonl` (prompt, completion, task, language), `lexicon.json` | SFT training, RL prompts (`build_rl_prompts`) |
| SFT | checkpoint | evaluator data (generations), RL start model and reference |
| evaluator | run directory with `best.pt` and config | RL reward (`csw.rl.evaluator`) |
| RL prompts | prompt pool with task, language, pin, source, must-keep items | RL |

## 4. Prompt and completion formats

All tags are single vocabulary tokens (tokenizer slots that were unused are renamed in `make_base`).

Prompts (`csw.sft.fmt2`), with `{iso}` the three-letter language code and `{switch}` the pinned switch-language code:

```
eng -> m_text   Translate the text to {Name}:  {english}           (two spaces after the colon)
eng -> hm       <hm> <{iso}>+<{switch}>:  {english}
m_text -> hm    <hm> <{iso}>+<{switch}>:  {m_text}
eng -> ood      <ood> <{iso}>:  {english}
m_text -> ood   <ood> <{iso}>:  {m_text}
hm -> ood       <ood> <{iso}>+<{switch}>:  {hm}
```

Task names used by the code: `eng_m`, `eng_hm`, `m_hm`, `eng_ood`, `m_ood`, `hm_ood`.

Completions are plan-first: the language tags commit to base and switch language, then the plan (`devices`), then the text:

```
m_text   {iso} {translation}                                        (plain code word, one space)
hm       <{iso}><{switch}><dev>kept phrase one | kept phrase two</dev> {code-switched text}
ood      <{iso}><dev>idiom one | idiom two</dev> {informal native text}
```

An optional `<draft> ... </draft>` block may precede the final text; it is cut out before anything is measured and has its own length cap. The chat wrapper is the Gemma turn format: the completion is followed by `<end_of_turn>`.

Evaluator input (`csw.evaluator.qe_lib`): `[CLS] task | language | pin  source [SEP] target [SEP] devices [SEP]`, with the devices dropped from a fraction of training examples so the same model can score text whose plan is unknown. Output: 27 numbers on the judge scale (1-5).

Judge rows and results: one JSON object per request with `rid`, `judge` (`<name>@v13`), `reply` (scores, typed errors, optional span edits `ed`/`fx`), `usage`, `schema_errors`.

## 5. Configuration

Nothing is stored in the repository. Paths and secrets come from the environment:

| variable | meaning | default |
|---|---|---|
| `CSW_DATA_ROOT` | directory with `seeds/`, `anchors/`, `exemplars/`, `work/`, `out*/` | current directory |
| `CSW_ENV_FILE` | `KEY=VALUE` file read into the environment (existing variables win) | `<data root>/.env` |
| `CSW_PAPER_DIR` | directory with the judge prompt card and its JSON schema | `<data root>/paper` |
| `CSW_WORK` | work directory of the self-training orchestrator | `work` |
| `CSW_PYTHON` | interpreter used by the orchestrator | current interpreter |
| `CSW_COMET_PYTHON`, `CSW_QE_LOG` | interpreter and log of the COMET worker process | current interpreter, `qe_worker.log` |
| `CSW_RETOK_DIR`, `CSW_TOKENIZER_JSON`, `CSW_TOKENIZER_SNAPSHOTS`, `CSW_HF_CACHE` | locations of tokenised pretraining files, tokenizer files and Hub caches | relative paths, see module docstrings |

API credentials are read from the environment only: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `TOGETHER_API_KEY`, `DOUBLEWORD_API_KEY`, `HF_TOKEN`, `HF_WRITE_TOKEN`. Several modules are written as scripts that execute when imported and are not meant to be imported for their definitions.

## 6. Packaging notes

* Layout: `src/csw/{datagen,judging,sft,evaluator,rl,vendor}`, `scripts/` (shell drivers for generation rounds and the serving wrapper), `tests/`, `docs/`.
* Optional dependency groups per stage are declared in `pyproject.toml`; the GPU-only groups (`sft-gen`, `rl-gen`) pull in vLLM.
* The tests that need a GPU, a network connection or private data are skipped with a reason.
