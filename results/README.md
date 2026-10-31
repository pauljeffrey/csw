# Results package

Results only: tables, figures and logs behind the paper (`../paper/paper.pdf`). No weights, data or usage instructions.

| folder | contents |
|---|---|
| `evaluator/` | agreement of the final evaluator with the LLM judges (`cmp_std.*`, `cmp_strict.*`), gold-row comparison (`pilot.txt`), low-end and corruption-chain checks (`lowend.txt`, `chains.txt`), comparison with SSA-COMET and AfriCOMET on 20,204 human-rated segments (`ssa_full_comparison_*.md`, `regression_final.json`), and the internals analysis (`internals/`) |
| `sft_ablations/` | the SFT ablation report (start model, depth, glossed devices, English draft) with its raw result files |
| `judges/` | LLM-judge validation against human ratings, with costs |
| `rl/logs/` | per-step training logs and periodic monitor logs of the RL runs (run 1-2: evaluator-led reward; runs 4-7: rule-anchored reward; runs 6-7 are split into resumed segments); `rl/sft_test_chrf/` the SFT-test chrF++ of the SFT model and of RL run 2 |
| `test_set/` | evaluator and rule scores of the SFT model, RL best and RL latest on the full SFT test set (per task, per item, rule penalties, paired differences) and chrF++ of each system |
| `samples/` | best-of-8 with the evaluator as selector: per-task summaries and per-item scores of the nine candidates |
| `judge_flash_v2_full/` | Gemini 3.8 Flash grades of the full test set for four systems with the standard prompt (analysis tables per task, criterion and language, scores, diffs) |
| `judge_argon_v2/` | Gemini 4 Argon cross-check on 1,000 stratified items (same prompt) |
| `judge_flash_v2/` | the earlier stratified Flash run, the judge prompt and the old strict prompt on a 300-item subset |
| `figures/` | the paper's figures (1-11) |

Notes
- Reward definitions changed between runs (see the paper, section 6); monitor curves are not comparable across those changes.
- Run 3 and the stale restart segments of run 6 are omitted: they were stopped after a few steps or resumed an older policy.
- Held-out reference chrF++ is logged from run 6 segment 2 onward only (the monitor file was missing from earlier segments).
