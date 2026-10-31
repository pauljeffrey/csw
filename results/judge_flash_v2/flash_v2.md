# Gemini 3.8 Flash, standard v2 prompt: 3999 graded of 4000 requests (unparseable 1); tokens in 3,046,777 out 552,057; batch cost $2.18

## Mean criterion average (1-5) per system and task, and paired difference to SFT (95% bootstrap CI over items)

| task | SFT | RL best | RL latest | generator reference | RL best - SFT | RL latest - SFT | reference - SFT |
|---|---|---|---|---|---|---|---|
| eng_hm | 3.581 | 3.572 | 3.591 | 4.438 | -0.009 [-0.098, +0.078] | +0.010 [-0.073, +0.092] | +0.858 [+0.753, +0.960] |
| m_hm | 3.476 | 3.256 | 3.242 | 4.607 | -0.218 [-0.316, -0.117] | -0.235 [-0.338, -0.131] | +1.135 [+1.024, +1.250] |
| eng_ood | 2.315 | 2.310 | 2.338 | 3.672 | -0.005 [-0.087, +0.073] | +0.023 [-0.054, +0.097] | +1.357 [+1.232, +1.472] |
| m_ood | 2.385 | 2.458 | 2.434 | 3.695 | +0.072 [-0.017, +0.159] | +0.049 [-0.043, +0.144] | +1.310 [+1.189, +1.418] |
| hm_ood | 2.271 | 2.382 | 2.350 | 3.721 | +0.111 [+0.025, +0.197] | +0.079 [-0.006, +0.167] | +1.450 [+1.325, +1.568] |
| all | 2.805 | 2.795 | 2.791 | 4.027 | -0.010 [-0.051, +0.031] | -0.015 [-0.055, +0.025] | +1.222 [+1.168, +1.276] |

## Per criterion (means over items), all code-switching tasks and all idiom tasks

| hm criterion | SFT | RL best | RL latest | generator reference | null share |
|---|---|---|---|---|---|
| meaning | 3.236 | 3.078 | 3.040 | 4.675 | 0.00 |
| base_language | 3.419 | 3.337 | 3.355 | 4.290 | 0.00 |
| switch_language | 3.892 | 3.862 | 3.835 | 4.782 | 0.00 |
| language_compliance | 4.794 | 4.825 | 4.785 | 4.902 | 0.00 |
| switching_naturalness | 3.000 | 2.763 | 2.797 | 4.322 | 0.00 |
| overall | 2.830 | 2.618 | 2.685 | 4.165 | 0.00 |

| ood criterion | SFT | RL best | RL latest | generator reference | null share |
|---|---|---|---|---|---|
| meaning | 1.863 | 1.983 | 1.963 | 3.457 | 0.00 |
| language_correctness | 2.343 | 2.422 | 2.425 | 3.608 | 0.00 |
| idiom_use | 1.587 | 1.505 | 1.490 | 3.440 | 0.00 |
| naturalness | 1.758 | 1.778 | 1.780 | 3.517 | 0.00 |
| language_compliance | 4.648 | 4.835 | 4.812 | 4.820 | 0.00 |
| overall | 1.742 | 1.775 | 1.775 | 3.333 | 0.00 |

## Head to head on the overall criterion (ties excluded)

| task | RL latest better | tie | SFT better |
|---|---|---|---|
| eng_hm | 0.26 | 0.47 | 0.27 |
| m_hm | 0.18 | 0.49 | 0.33 |
| eng_ood | 0.19 | 0.64 | 0.17 |
| m_ood | 0.26 | 0.54 | 0.20 |
| hm_ood | 0.25 | 0.57 | 0.17 |

## Prompt sensitivity: old strict compact prompt vs standard v2 on the same items (1198 graded)

| task | system | v2 criterion mean | old prompt dimension mean (H1-H9 or O1-O7) |
|---|---|---|---|
| eng_hm | SFT | 3.456 | 3.129 |
| eng_hm | RL best | 3.494 | 3.276 |
| eng_hm | RL latest | 3.600 | 3.246 |
| eng_hm | generator reference | 4.361 | 3.805 |
| m_hm | SFT | 3.569 | 3.235 |
| m_hm | RL best | 3.297 | 3.123 |
| m_hm | RL latest | 3.242 | 3.123 |
| m_hm | generator reference | 4.583 | 3.967 |
| eng_ood | SFT | 2.319 | 2.365 |
| eng_ood | RL best | 2.308 | 2.544 |
| eng_ood | RL latest | 2.369 | 2.590 |
| eng_ood | generator reference | 3.661 | 3.531 |
| m_ood | SFT | 2.361 | 2.310 |
| m_ood | RL best | 2.481 | 2.587 |
| m_ood | RL latest | 2.456 | 2.600 |
| m_ood | generator reference | 3.581 | 3.395 |
| hm_ood | SFT | 2.285 | 2.267 |
| hm_ood | RL best | 2.378 | 2.537 |
| hm_ood | RL latest | 2.408 | 2.476 |
| hm_ood | generator reference | 3.786 | 3.572 |

Spearman between the two prompts across graded (item, system) pairs: 0.843 (n=1198).