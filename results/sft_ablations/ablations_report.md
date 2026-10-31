# csw (code-switching + idiom model, 270M-335M): format, start-point and depth ablation on a 20% subsample (2026-10-28/29)

Everything here was run on one GPU, one seed per arm, on a 20% stratified subsample of the csw SFT build. Raw result files are in `data/`.

## 0. Summary with the evidence level of every claim

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | Starting from the translation-tuned model beats starting from plain `google/gemma-3-270m` | val loss 1.47 vs 1.91; chrF 35.1/38.6 vs 31.1; 20.7% looping outputs vs 0.7-9.3% | **Established** (large gap; the chrF comparison mixes two decoders but the validation loss does not) |
| 2 | Adding 6 interleaved identity-initialised layers (E, 301M) beats adding 12 (F, 335M) | E2 vs F, same data, same recipe, same decoder (vLLM), same 2,688 rows: 38.57 vs 37.70, 95% CI of the gap [+0.57, +1.17]; E wins all 5 tasks; also lower val loss in the first run (1.465 vs 1.513, both at 32,768 tokens per pass) | **Established** |
| 3 | Adding 6 layers (E) beats the 18-layer baseline (A) | **Refuted at this data size.** Clean test A2 vs E2 (same data, recipe, micro-batch, vLLM decoder, same 2,688 rows): A2 38.72 vs E2 38.57, CI of (A2 - E2) [-0.16, +0.45]; A2 has fewer loops (2.2% vs 3.0%). Earlier HF-decoded A vs E was also a tie (35.09 vs 35.09); E's loop advantage there (0.7% vs 9.3%) was a decoder effect | **Not supported: depth up-scaling gave no gain over the baseline at 60M training tokens** |
| 4 | Glossed devices (B) hurt | -1.1 to -2.1 chrF on the three ood tasks, more English leak | Established (above the ~1-1.5 chrF noise floor on 3 tasks) |
| 5 | An English draft helps only eng->ood | draft on: eng_ood +0.5 (greedy) / +0.8 (sampled), m_ood -2.1 / -2.0, hm_ood -0.4 / -0.3 | Established for m_ood; the eng_ood gain is inside the noise floor |
| 6 | chrF noise floor between two decode runs of the same checkpoint | 1-1.6 chrF (same prompts in two decode runs) | Measured; **differences under ~1.5 chrF from a single run are not reliable** |

**Claim 3 was tested cleanly and is not supported** (arm A2, section 6.6). An earlier partial result (first 1,728 shared rows) showed E +1.45 chrF over A; it disappeared when the shared set grew to 2,688 rows (section 6.5), and the clean A2-vs-E2 comparison is a tie with the baseline nominally ahead.

## 1. Task and data

* Model task (csw): five generation tasks over 37 African languages, plus an English->African translation replay task (`eng_m`). Prompts `<hm> <yor>+<eng>:  text` (code-switch), `<ood> <yor>:  text` and `<ood> <yor>+<eng>:  hm text` (informal native text with idioms); completion `<base>[<switch>]<dev>device one | device two</dev> text`. Every tag is one vocabulary token (52 renamed `<unusedN>` slots; vocabulary stays 262,144).
* Tasks: eng_hm, m_hm, eng_ood, m_ood, hm_ood (+ eng_m replay in training only). Generated text: `google/gemma-4-31b-it` (FP8) on four task cards, filtered by rules and by two LLM judges (Gemini 3.8 Flash, GPT-5.6 luna; Batch API) on about 22% of rows; judge-corrected copies added.
* Full build `csw_sft_v3`: train 832,421 rows, val 17,399, test 6,230 (curated: judged rows with no major or critical error; no English source sentence shared with train).
* **Ablation subsample** (`csw.build_ablation --frac 0.2`): stratified by task x language: **train 166,484 rows (about 59.9 M tokens), val 3,488, test 6,230** (the full curated test). 103,048 of the train rows are ood tasks.
* Three formats built from the same rows (only the ood tasks differ; hm, eng_m and replay are identical):
  * **A standard:** `<yor><dev>idiom one | idiom two</dev> text`
  * **B glossed:** `<dev>idiom one (expression: meaning) | ...</dev>` (the generator's own English glosses)
  * **C draft:** a random half of the ood examples carry the flag `<draft> ` in the prompt and `<draft>English draft</draft>` before `<dev>` in the completion (the draft = the generator's `eng_initial`); the other half equal A, so one model learns both modes. Test files `test_on` (flag on every ood prompt) and `test_off`.

## 2. Arms

| arm | start model | layers / params | format | trained on | micro-batch tokens | freeze phase | decoder |
|---|---|---|---|---|---|---|---|
| A | translation-tuned Gemma-3-270M + tags | 18 / 268.1M | standard | 20% | 32,768 | - | HF |
| B | same | 18 / 268.1M | glossed | 20% | 32,768 | - | HF |
| C | same | 18 / 268.1M | draft (half) | 20% | 32,768 | - | HF |
| D | plain `google/gemma-3-270m` + same tags | 18 / 268.1M | standard | 20% | 16,384 | - | vLLM |
| E | translation-tuned, **+6 layers** | 24 / 301.5M | standard | 20% | 32,768 | first 120 steps | HF (stopped early, 2,688 rows) |
| F | translation-tuned, **+12 layers** | 30 / 335.0M | standard | 20% | 32,768 | first 120 steps | vLLM |
| E2 | arm E retrained (checkpoints of A and E were deleted after scoring) | 24 / 301.5M | standard | 20% | **16,384** | first 120 steps | vLLM |
| A2 | arm A retrained under E2's settings | 18 / 268.1M | standard | 20% | 16,384 | - | vLLM |

* **Depth up-scaling** (`csw.upscale`): after every 3rd original layer, 1 (E) or 2 (F) new layers are inserted (new indices E: 3, 7, 11, 15, 19, 23). A new layer is a copy of the layer before it with the attention output projection and the MLP down projection set to zero. Both sub-blocks end in a norm before the residual add, so the new layers add exactly nothing at the start: the up-scaled model's logits equal the source model's (maximum difference 0.0, verified). New layers use sliding-window attention; original layers keep their own type (including the full-attention ones). **Freeze phase:** for the first 120 of about 460 steps only the new layers train (33.4M of 301.5M parameters for E); then everything trains.
* The micro-batch size does not change the optimizer step (262,144 tokens in every arm; the loss is normalised by total tokens per update), only the grouping. The E-vs-F difference is already visible with equal micro-batches in the first run (val loss), so it is not caused by the 16,384/32,768 difference; this has not been measured directly.

## 3. Training recipe (identical for all arms; `csw.train_sft`)

Full fine-tuning; fp32 weights with bf16 autocast; AdamW (lr 5e-5, betas 0.9/0.95, weight decay 0.01, fused); warm-up 3% then cosine decay to 10% of the peak; gradient clip 1.0; 2 epochs = 460-464 optimizer steps; loss only on completion tokens; token-budget batches (sorted by length, shuffled); maximum length 2,048; chat template `<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`; validation loss every 100 steps, best checkpoint kept.

## 4. Evaluation protocol

* **Decoders.** HF `generate` (bf16, left padding, batch 96 sorted by prompt length, 10 processes) for A, B, C, E. vLLM 0.32 (bf16, `max_model_len` 2,048) for D, F, E2. Same prompts and sampling in both: sample 0 greedy (repetition penalty 1.0; 1.05 harms this format), sample 1 temperature 0.7 / top-p 0.95, max 700 new tokens, stop at `<end_of_turn>`. vLLM was checked against HF on the 30-layer model (mean absolute difference of per-token log-probabilities 0.064 over 1,100 tokens, the same inside and beyond the 512-token sliding window, so the layer layout is handled correctly).
* **Metric.** chrF++ (sacreBLEU, word order 2) of the **final text only** (language tags, `<draft>..</draft>` and `<dev>..</dev>` removed from hypothesis and reference). Format ablation: corpus chrF per task x language, macro-averaged over languages. Architecture comparison: sentence-level chrF++, mean per task, then mean over the 5 tasks ("score").
* **Other columns.** `loop%` = share of greedy outputs whose final text has a repeated-4-gram ratio below 0.6; `noclose%` = no `</dev>` in the output; `fmt_ok` = exactly one `<dev>` block with at least 2 devices and non-empty text; `len` = mean generation/reference length ratio in characters; `dev_in` = share of the generation's own devices that occur in its text; `en_rate` = English function-word share minus the reference's.
* **Significance.** Paired bootstrap (1,000 resamples, stratified by task) of (best - arm). Rule for picking the smallest adequate model: the smallest arm whose gap CI includes 0 and whose gap is under 1.0 chrF.
* **Shared rows.** Arm E's generation was stopped early, so all architecture comparisons use the 2,688 rows E had finished (`test_Edone.jsonl`). **These are not a random sample:** each shard writes shortest prompts first. 34 languages, very uneven (nde 11.6%, mlg 9.1%, amh 6.5%, ssw 6.2%, afr 6.0%, pcm 5.2%, tum 5.1% ... bam 3 rows, twi 7, run 11, lin 11, zul 12; top five languages 39.3%). Tasks: m_ood 23.4%, eng_ood 22.6%, m_hm 18.7%, eng_hm 18.2%, hm_ood 17.0%. Treat per-language numbers as noise and report pooled numbers only.

## 5. Format ablation (A / B / C), full curated test (6,230 rows), HF decoder, 34 languages, macro over languages

chrF++ of the final text; greedy sample (s0) and sampled (s1):

| task | A s0 | A s1 | B s0 | B s1 | C draft-off s0 | C draft-off s1 | C draft-on s0 | C draft-on s1 |
|---|---|---|---|---|---|---|---|---|
| eng_hm | 42.51 | 40.72 | 42.31 | 40.57 | 42.12 | 40.95 | (no draft in hm prompts) | |
| m_hm | 48.68 | 46.88 | 46.72 | 46.26 | 49.53 | 48.64 | (no draft in hm prompts) | |
| eng_ood | 30.84 | 32.00 | 28.77 | 30.95 | 30.53 | 31.68 | **31.35** | **32.77** |
| m_ood | 37.09 | 37.30 | 35.80 | 37.09 | 36.60 | 36.73 | 35.03 | 35.30 |
| hm_ood | 33.12 | 33.53 | 31.98 | 33.92 | 32.22 | 33.23 | 32.74 | 33.25 |

Secondary columns (greedy, s0): format validity `fmt_ok` A 0.90-0.95, B 0.93-0.96, C draft-on 0.96-0.97 on ood; length ratio A 1.31-1.61, B 1.51-1.89, C draft-on 1.06-1.32; English leak (`en_rate`) about 0 for A, +0.015 to +0.025 for B on ood.
The hm rows of the draft-on test have identical prompts to draft-off; their differences (e.g. eng_hm 42.12 vs 43.70) are decode-to-decode noise and give the noise floor (1-1.6 chrF).

## 6. Architecture and start-point ablation, 2,688 shared rows

### 6.1 All arms (`compare_all.txt`); gap = 95% CI of (best - arm), best = E2

| arm | params | start | decoder | score | eng_hm | eng_ood | hm_ood | m_hm | m_ood | loop% | noclose% | fmt_ok | len | gap to best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 268M | translation-tuned | HF | 35.09 | 42.19 | 27.94 | 27.46 | 47.20 | 30.66 | 9.3 | 7.2 | 0.885 | 1.63 | [+2.90, +4.02] |
| E | 301M (+6) | translation-tuned | HF | 35.09 | 40.68 | 28.53 | 28.48 | 47.19 | 30.58 | 0.7 | 10.5 | 0.860 | 0.99 | [+3.10, +3.88] |
| F | 335M (+12) | translation-tuned | vLLM | 37.70 | 43.91 | 29.95 | 29.70 | 52.08 | 32.84 | 2.9 | 0.1 | 0.991 | 1.12 | [+0.57, +1.17] |
| **E2** | 301M (+6) | translation-tuned | vLLM | **38.57** | 45.57 | 30.96 | 30.32 | 52.93 | 33.09 | 3.0 | 0.1 | 0.986 | 1.18 | best |
| D | 268M | plain gemma-3-270m | vLLM | 31.06 | 32.60 | 21.22 | 22.94 | 49.39 | 29.16 | 20.7 | 0.9 | 0.989 | 2.00 | [+7.11, +7.92] |

**Do not read A and E against F, E2 and D.** The HF arms (A, E) and the vLLM arms differ in decoder as well as architecture; the HF arms show 7-10% missing `</dev>` and about 0.86-0.89 `fmt_ok`, the vLLM arms 0.1% and 0.99. Within a decoder family the comparisons are clean: A vs E (HF), F vs E2 vs D (vLLM).

### 6.2 The fair test: E2 vs F (same data, recipe, decoder, rows)

| arm | score | eng_hm | eng_ood | hm_ood | m_hm | m_ood | gap to best (CI95) |
|---|---|---|---|---|---|---|---|
| E2 (301M) | **38.57** | 45.57 | 30.96 | 30.32 | 52.93 | 33.09 | - |
| F (335M) | 37.70 | 43.91 | 29.95 | 29.70 | 52.08 | 32.84 | [+0.57, +1.17] |

E2 is better on all five tasks and smaller. Loop and format rates are the same. The only recipe difference is the micro-batch (16,384 for E2, 32,768 for F; same optimizer step).

### 6.3 A vs E, same HF decoder (`compare_A_vs_E.txt`)

| arm | score | eng_hm | eng_ood | hm_ood | m_hm | m_ood | loop% | noclose% | len | gap to best (CI95) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 35.09 | 42.19 | 27.94 | 27.46 | 47.20 | 30.66 | 9.3 | 7.2 | 1.63 | [-0.46, +0.44] |
| E | 35.09 | 40.68 | 28.53 | 28.48 | 47.19 | 30.58 | 0.7 | 10.5 | 0.99 | best |

**Tie on chrF** (identical score, interval centred on 0). E is better on eng_ood (+0.6) and hm_ood (+1.0), equal on m_hm and m_ood (-0.1), and worse on eng_hm (-1.5); E has 13 times fewer looping outputs and the length ratio is 0.99 instead of 1.63, but more outputs end without a closing `</dev>`.

### 6.4 Validation loss (same 3,488-row validation set for every standard-format arm; independent of the decoder)

| arm | best val loss |
|---|---|
| A | 1.4698 |
| E (32,768 per pass) | 1.4648 |
| **E2 (16,384 per pass)** | **1.4514** |
| F | 1.5130 |
| D (plain gemma) | 1.9109 |
| B (glossed targets; different target text, not comparable) | 1.5295 |
| C (draft targets; different target text, not comparable) | 1.4442 |

A to E is 0.005: single seed, not distinguishable from noise.

### 6.5 History of the A-vs-E result (why claim 3 is not established)

First partial comparison on the 1,728 rows E had finished at that moment: A 33.23 vs E 34.68 (+1.45; E ahead on all five tasks; loops 6.6% vs 0.7%). On the final 2,688 shared rows the scores are equal (35.09 vs 35.09). The early advantage came from an unrepresentative subset; the loop reduction is the only effect that survived.

### 6.6 The clean test of claim 3: A2 vs E2 vs F (same data, recipe, micro-batch 16,384 for A2/E2 and 32,768 for F, vLLM, same 2,688 rows; `compare_A2.txt`)

| arm | params | score | eng_hm | eng_ood | hm_ood | m_hm | m_ood | loop% | noclose% | fmt_ok | len | gap to best (CI95) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A2** | 268M | **38.72** | 45.77 | 31.22 | 30.48 | 52.87 | 33.24 | 2.2 | 0.1 | 0.991 | 1.09 | best |
| E2 | 301M | 38.57 | 45.57 | 30.96 | 30.32 | 52.93 | 33.09 | 3.0 | 0.1 | 0.986 | 1.18 | [-0.16, +0.45] |
| F | 335M | 37.70 | 43.91 | 29.95 | 29.70 | 52.08 | 32.84 | 2.9 | 0.1 | 0.991 | 1.12 | [+0.72, +1.31] |

A2 and E2 are statistically tied (the interval includes 0; A2 is nominally ahead on four of five tasks and has fewer loops); F is clearly worse. By the pre-set rule (smallest arm within noise of the best) the winner is A2. **Depth up-scaling by interleaved identity-initialised layers brought no benefit over the baseline at 60M training tokens and 460 steps; 12 new layers hurt.** This is a single-seed result on 20% of the data; whether more layers pay off with the full 300M-token build is not tested. The loop reduction seen for E under the HF decoder (section 6.3) is therefore a decoder effect, not an architecture effect.

## 7. What is confounded or missing

* Single seed per arm, 20% data, 2 epochs; the noise floor between two decode runs is 1-1.6 chrF.
* The shared 2,688 rows are the shortest prompts of each shard and language-skewed (section 4).
* A, B, C, E were decoded with HF and their checkpoints were deleted after scoring; they cannot be re-decoded with vLLM.
* E2 and D used 16,384 tokens per pass, the others 32,768 (same optimizer step; not measured).
* The depth effect was only compared against one position pattern (new layer after every 3rd original layer, sliding attention, identity start) and one freeze length (120 steps); no other pattern, no global-attention new layers, no longer freeze.
* The full-data picture is unknown: all arms were trained on 60M tokens, the full build has 300M.

## 8. Reusing this as an ablation

* **Done (arm A2):** A retrained on the same 20% data with E2's settings, decoded with vLLM on the same rows: a tie with E2 (section 6.6).
* **For the translation work:** copy the recipe in sections 2-4 (`csw.upscale --every 3 --per-group 1`, `csw.train_sft --freeze-steps 120`), keep the checkpoints of all arms, decode every arm with the same decoder, and use a random, language-stratified evaluation set instead of an early-stopped one. Use sentence-level chrF++ with the paired bootstrap above; report loops (`loop%`) and the validation loss next to chrF, because chrF alone missed the loop effect here.

## 9. Files

* Results: `data/` (`compare_all.json/.txt`, `compare_A_vs_E.*`, `compare_E2_vs_F.*`, `compare.*`, per-arm `A_test.*`, `B_test.*`, `C_test_off.*`, `C_test_on.*`, `D_test_Edone.*`, `F_test_Edone.*`, `E2_test_Edone.*`, `ablation_info.json`).
* Decontamination: no benchmark dev/devtest/test text in training; the human translation replay comes from a cleaned human translation set.
