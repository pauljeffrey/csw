# test_set: human preference study pool

Private. Nothing here has been seen by native speakers yet. Automatic scores below are frozen before any human vote is collected; do not refit a judge to the votes and then report agreement on the same votes.

## What is in it

8,300 items over 37 base languages, each with up to 12 candidate outputs (mean about 10), and 29,938 prepared pairs.

- `items.jsonl.gz`: one row per item. Fields: `id`, `origin`, `task`, `lang`, `switch`, `english` (the text shown as source), `prompt`, `candidates`. Each candidate has `text`, `devices`, `tags`, `systems`, `greedy`, `ev` (evaluator score, 1 to 5) and, where it exists, `flash_overall` (Gemini Flash grade, 1 to 5, keyed by system).
- `pairs.jsonl.gz`: the pairs to show raters (`item`, `kind`, `a`, `b`, `ev_gap`, `swap`).
- `attention_checks.jsonl`: a candidate against a word-shuffled copy of itself. The shuffled one should always lose.
- `stats.json`: counts per language, task and pair kind, evaluator deciles.

### Origins

- `test_set` (6,225 items): the curated test set from the project (no English source shared with training), five tasks: `eng_hm`, `m_hm`, `eng_ood`, `m_ood`, `hm_ood`. Candidates are the human-style gold reference, the SFT output, the RL-last output (the three have Flash grades; identical outputs are merged and keep all system names), plus one greedy and up to seven sampled candidates from the RL-last model.
- `pool_low_range_boost` (300 items): extra pool items for pcm, gpe and wes (100 each), drawn from the lowest-scoring 30% of their rows. The evaluator scores nearly everything in these three languages at 4.2 or above, so without them a judge could not be tested on weak outputs there. Their score spread (std) went from 0.20 to 0.26 up to 0.34 to 0.54.
- `pool` and `test_set_eng_m`: items from `data/` (new English chats and stories; translation by rl2c, code-switching and idioms by RL-last) and rl2c translations of the test set's English sources. These have evaluator scores only, no Flash grades and no reference.

Sampling settings: translation T 0.8, min-p 0.1; code-switching and idioms T 1.0, min-p 0.1. Candidate 0 is always greedy.

## How the items were chosen

1. Every test-set item is in (they already carry Flash grades).
2. Languages are filled evenly from the pool up to a common level (142 items), stratified by task and by the mean evaluator score of the row so the whole score range is covered. Where the test set already has more than 142 items for a language it keeps all of them, so the final counts are uneven (see `stats.json`; the most items are afr, amh, swh, som, hau, mlg).
3. Pairs per item: greedy against a random sampled candidate; the highest against the lowest evaluator score; two neighbours near the median; and, for test-set items, the gold reference against the RL-last output. Pairs are not restricted to high scores, so a judge that ranks nothing will show it.

## Protocol to follow when collecting votes

- Randomise left and right on every view (`swap` is only a seeded default). Never show system names, scores or the pair kind.
- Show a pair only to a rater whose native or near-native language is the item's base language. Record rater id, native language(s), self-reported level in the base and switch language, and time taken.
- Insert an attention check about every 25 pairs. Drop raters who fail more than one in five; report how many were dropped.
- Allow "tie / both bad". Report it, do not force a choice.
- Get at least two raters on a subset of pairs per language so inter-annotator agreement (Krippendorff alpha or Cohen's kappa) can be reported per language. That agreement is the ceiling any automatic judge can be compared against.

## What will be reported

Headline numbers: (1) pairwise win-rate, the share of pairs where the automatic judge prefers the same candidate as the human majority; (2) agreement of each signal with human choice, for the evaluator, Flash, chrF++ against the reference where one exists, and the rule score; (3) human agreement per language as the ceiling. Framing: how far automatic signals alone get, measured against humans.

Minimum for a language to be reported on its own: at least 300 pair judgments from at least 3 distinct raters, and at least 100 pairs with two or more raters. Languages below that are pooled into a single "other" row and not interpreted individually. I cannot say now which languages will clear the bar: every language has at least 400 prepared pairs, so what limits it is how many native raters show up. The table of votes per language will be published with the results.

## Known limits

Flash grades exist only for test-set items and only for the reference, SFT and RL-last outputs. The evaluator is weak at ranking outputs of one prompt (54% agreement with Flash), which is part of what this study measures. Nothing was checked by native speakers. ful, gpe and nde have no translation (rl2c) items. Source texts keep their own licences; outputs are subject to the Gemma terms; research use only.
