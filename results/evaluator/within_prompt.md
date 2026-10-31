# Evaluator signal against Flash (overall criterion), 18629 outputs, 6222 prompts

| view | evaluator | reward |
|---|---|---|
| pooled over all outputs, Spearman | -0.194 | -0.222 |
| inside one prompt (3 systems), Pearson | +0.110 | +0.004 |

Pairwise: when Flash prefers one output the evaluator prefers the same one 54.3% of the time (8195 pairs; 10362 pairs tied for Flash were skipped).

| task | evaluator mean | Flash overall mean | pooled Spearman in task | inside-prompt Pearson |
|---|---|---|---|---|
| eng_hm | 3.04 | 3.00 | -0.003 | +0.088 |
| eng_ood | 4.08 | 1.84 | +0.355 | +0.126 |
| hm_ood | 3.95 | 1.70 | +0.212 | +0.130 |
| m_hm | 3.08 | 2.60 | +0.099 | +0.039 |
| m_ood | 4.08 | 1.88 | +0.284 | +0.163 |

Calibration fitted on the SFT outputs (20 evaluator quantile bins -> mean Flash overall), applied to every system:

| system | evaluator | Flash predicted from evaluator | Flash actual | actual - predicted |
|---|---|---|---|---|
| sft | 3.528 | 2.264 | 2.264 | +0.000 |
| rl_best | 3.750 | 2.286 | 2.152 | -0.134 |
| rl_last | 3.729 | 2.296 | 2.147 | -0.149 |

On the SFT outputs one evaluator point is worth 0.03 Flash points; mean Flash minus evaluator -1.26. A constant offset cancels in GRPO (advantages are differences inside a group), so this bias is irrelevant to the update.

Sampled groups (RL best, 6225 prompts, 9 candidates each, no Flash grades): inside-prompt Pearson between evaluator and chrF++ +0.269.
