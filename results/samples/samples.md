# Best-of-N with the evaluator as selector (corpus chrF++ per task, micro; evaluator mean over format-valid outputs)

| system | task | strategy | chrF++ | evaluator | format fail |
|---|---|---|---|---|---|
| sft | eng_hm | greedy | 55.14 | 3.013 | 0.020 |
| sft | eng_hm | random_sample | 52.47 | 3.000 | 0.021 |
| sft | eng_hm | evaluator_pick | 52.78 | 3.132 | 0.003 |
| sft | eng_hm | oracle_chrf | 58.49 | 3.009 | 0.003 |
| sft | eng_ood | greedy | 38.70 | 3.867 | 0.001 |
| sft | eng_ood | random_sample | 38.91 | 4.009 | 0.000 |
| sft | eng_ood | evaluator_pick | 39.07 | 4.259 | 0.000 |
| sft | eng_ood | oracle_chrf | 42.74 | 4.018 | 0.000 |
| sft | hm_ood | greedy | 39.65 | 3.715 | 0.000 |
| sft | hm_ood | random_sample | 39.89 | 3.890 | 0.001 |
| sft | hm_ood | evaluator_pick | 39.97 | 4.149 | 0.000 |
| sft | hm_ood | oracle_chrf | 43.59 | 3.906 | 0.000 |
| sft | m_hm | greedy | 61.93 | 3.045 | 0.017 |
| sft | m_hm | random_sample | 59.97 | 3.033 | 0.018 |
| sft | m_hm | evaluator_pick | 60.17 | 3.160 | 0.001 |
| sft | m_hm | oracle_chrf | 65.82 | 3.036 | 0.001 |
| sft | m_ood | greedy | 40.89 | 3.893 | 0.001 |
| sft | m_ood | random_sample | 40.77 | 3.986 | 0.000 |
| sft | m_ood | evaluator_pick | 40.94 | 4.247 | 0.000 |
| sft | m_ood | oracle_chrf | 44.67 | 4.014 | 0.000 |
| rl_last | eng_hm | greedy | 54.54 | 3.045 | 0.003 |
| rl_last | eng_hm | random_sample | 52.77 | 3.039 | 0.003 |
| rl_last | eng_hm | evaluator_pick | 52.80 | 3.163 | 0.001 |
| rl_last | eng_hm | oracle_chrf | 57.82 | 3.044 | 0.001 |
| rl_last | eng_ood | greedy | 38.60 | 4.173 | 0.000 |
| rl_last | eng_ood | random_sample | 37.89 | 4.203 | 0.000 |
| rl_last | eng_ood | evaluator_pick | 37.82 | 4.346 | 0.000 |
| rl_last | eng_ood | oracle_chrf | 41.57 | 4.196 | 0.000 |
| rl_last | hm_ood | greedy | 40.11 | 4.054 | 0.000 |
| rl_last | hm_ood | random_sample | 39.30 | 4.101 | 0.000 |
| rl_last | hm_ood | evaluator_pick | 39.28 | 4.244 | 0.000 |
| rl_last | hm_ood | oracle_chrf | 42.81 | 4.090 | 0.000 |
| rl_last | m_hm | greedy | 61.21 | 3.099 | 0.001 |
| rl_last | m_hm | random_sample | 60.17 | 3.095 | 0.003 |
| rl_last | m_hm | evaluator_pick | 59.78 | 3.226 | 0.000 |
| rl_last | m_hm | oracle_chrf | 64.89 | 3.098 | 0.000 |
| rl_last | m_ood | greedy | 41.37 | 4.165 | 0.000 |
| rl_last | m_ood | random_sample | 40.48 | 4.199 | 0.000 |
| rl_last | m_ood | evaluator_pick | 40.48 | 4.326 | 0.000 |
| rl_last | m_ood | oracle_chrf | 44.09 | 4.193 | 0.000 |