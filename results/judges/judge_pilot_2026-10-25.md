# LLM-judge validation pilot (J1 report, saved 2026-10-25)

Question: which Gemini judge to use, following the SSA-COMET paper (Li et al., EMNLP 2025, arXiv 2506.04557). `gemini-2.5-pro` returned 404 "no longer available to new users" (batch and standard) for this API key, so the stronger arm is `gemini-3.1-pro-preview` (Google's named successor; $2/$12 standard, $1/$6 batch per 1M tokens) against `gemini-3.8-flash` ($0.75/$3.75 standard until 2026-12-31, batch half). The exact 2.5-Pro reproduction was not possible.

## What was run
- 300 segments, identical for both judges: SSA-MTE eng-amh, eng-yor, eng-kin, eng-hau (50 each; test pool, train split from the public `McGill-NLP/SSA-MTE`, CC-BY-SA-4.0) and AfriMTE eng-swh, eng-ibo, eng-som, eng-xho (25 each; `masakhane/AfriMTE-WMT2024`, CC-BY-4.0). Seed 2026, 5 strata of the human z-score, equal counts per stratum. Human target = per-annotator z-score.
- Prompt: SSA-COMET Figure 8 (reference-based, score only), verbatim from the paper's PDF; 5-shot, one demonstration per adequacy interval of the training set. Deviations: AfriMTE demonstrations come from the same pair's other AfriMTE rows (z min-max scaled to 0-100; the paper does not describe this); SSA demonstration scores are raw DA 0-100; demonstration order is a seeded shuffle (paper silent); `[67] Most` written with a space.
- Temperature 0, thinking at each model's default, Batch API only, one run per judge. `max_output_tokens` 2300 (Pro) / 2200 (Flash); a 12-request probe measured thinking use first. Budget guard: worst-case cost before each submit (exact `count_tokens` + n x cap x output price + 5%), refused above USD 6; worst-case sum 5.91.
- Failures: 0 parse failures, 0 non-STOP finishes for both judges (strict, dropped and lenient variants identical).

## Cost (actual usage_metadata)
| judge | input tokens | output incl. thinking | cost |
|---|---|---|---|
| 3.1-Pro-preview | 307,364 | 204,190 | $1.533 |
| 3.8-Flash | 307,364 | 92,373 | $0.288 |
Total $1.82 of the $6.00 budget; nothing spent on 2.5-Pro. Thinking tokens: Pro mean 679 (max 1,315), Flash mean 306 (max 652). Per segment: Pro about $0.0051, Flash about $0.00096.

## Spearman vs human z-score (95% bootstrap CI over segments, 4,000 resamples)
| scope | 3.1-Pro | 3.8-Flash | SSA-COMET-MTL (ref) | paper Gemini-2.5-Pro MTE |
|---|---|---|---|---|
| amh | 0.540 [.28,.74] | 0.561 [.29,.77] | 0.587 | 0.605 |
| hau | 0.516 [.28,.70] | 0.612 [.39,.78] | 0.518 | 0.471 |
| kin | 0.486 [.22,.69] | 0.535 [.28,.74] | 0.616 | 0.528 |
| yor | 0.618 [.40,.78] | 0.639 [.40,.81] | 0.475 | 0.524 |
| swh | 0.818 | 0.857 | 0.866 | - |
| ibo | 0.528 | 0.529 | 0.520 | - |
| som | 0.317 | 0.513 | 0.700 | - |
| xho | 0.192 | 0.190 | 0.330 | - |
| SSA 4 pooled | 0.471 [.35,.58] | 0.537 [.41,.64] | 0.530 | - |
| AfriMTE 4 pooled | 0.438 [.25,.59] | 0.462 [.28,.62] | 0.635 | - |
| POOLED (300) | 0.454 [.35,.55] | 0.511 [.41,.60] | 0.536 | - |
| POOLED, per-pair means removed | 0.538 [.44,.62] | 0.584 [.49,.67] | 0.556 | - |
AfriCOMET-1.1-STL (ref) pooled 0.447; AfriCOMET-1.0-MTL (ref) 0.449. Pearson pooled Pro 0.566, Flash 0.610; demeaned 0.613 / 0.661.
Paired difference Pro minus Flash (same resamples): pooled Spearman -0.057 [-0.09, -0.02], Pearson -0.044 [-0.08, -0.01]; demeaned -0.046 [-0.08, -0.02]; SSA 4 pooled -0.065 [-0.11, -0.02]; AfriMTE 4 pooled -0.024 [-0.09, +0.04]; per pair every CI includes 0 except hau (-0.096 [-0.20, -0.01]) and som (-0.197 [-0.44, -0.03]). Pro minus SSA-COMET-MTL pooled -0.082 [-0.17, 0.00]. The two judges' scores correlate 0.94 (Spearman; computed by P11 from the segment file).

## What 300 segments can and cannot resolve
Per pair (n = 25-50) the SE of a difference is 0.04-0.2: per-pair rankings are noise (xho, som especially). Pooled over 300 the SE of Pro minus Flash is about 0.018 (gaps of about 0.04 detectable). Segments within a pair share source sentences across systems and were treated as independent, so CIs are somewhat optimistic. The pilot is not a 2.5-Pro reproduction (the paper's kik, luo, twi pairs were not sampled; 3.1-Pro within about 0.1 of the paper's 2.5-Pro numbers on the four shared pairs, well inside the n=50 CI).

## Recommendation
3.8-Flash is not worse than 3.1-Pro as a judge (slightly better pooled, about 5x cheaper). Both are at or below SSA-COMET-MTL on pooled Spearman; the learned metric is clearly better on som and xho and Flash roughly matches it elsewhere: the judge's value is independence from our reward, not higher accuracy. Use 3.8-Flash (Batch API) as the default judge. Validation covers the reference-based prompt on 11 languages with demonstration data (amh hau kik kin luo twi yor swh ibo som xho; kik, luo, twi not sampled); the reference-free prompt is not yet validated. Optional extension: 150 more segments per SSA pair and 25 per AfriMTE pair (700 segments, about $0.7 for Flash alone).

Files: scripts `gemma_mt/scripts/judge_pilot_{common,build,batch,analyze}.py`; outputs `reports/data/judge_pilot_2026-10-25/` (correlations.csv, paired_differences.csv, paper_reproduction_check.csv, sensitivity_parse.csv, judge_scores_by_segment.csv, cost_and_failures.csv, ledger.json, demo_ids.json, sample_ids.csv, raw_*.jsonl, prompt_example_pair_eng-yor.txt). Segment text, full prompts and datasets stayed in the scratchpad only. No API key appears in any repo file.
