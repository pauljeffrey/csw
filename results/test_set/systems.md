# Evaluator and rule scores of each system on the same test items (draft off)

| task | metric | sft | rl_best | rl_last |
|---|---|---|---|---|
| eng_hm | evaluator | 3.013 | 3.063 | 3.044 |
| eng_hm | reward | 0.122 | 0.285 | 0.283 |
| eng_hm | gate_rate | 0.020 | 0.004 | 0.003 |
| eng_ood | evaluator | 3.867 | 4.196 | 4.173 |
| eng_ood | reward | 0.473 | 0.717 | 0.708 |
| eng_ood | gate_rate | 0.001 | 0.000 | 0.000 |
| hm_ood | evaluator | 3.715 | 4.078 | 4.054 |
| hm_ood | reward | 0.470 | 0.717 | 0.704 |
| hm_ood | gate_rate | 0.000 | 0.000 | 0.000 |
| m_hm | evaluator | 3.044 | 3.104 | 3.099 |
| m_hm | reward | 0.141 | 0.312 | 0.315 |
| m_hm | gate_rate | 0.017 | 0.003 | 0.001 |
| m_ood | evaluator | 3.891 | 4.195 | 4.165 |
| m_ood | reward | 0.525 | 0.750 | 0.742 |
| m_ood | gate_rate | 0.001 | 0.000 | 0.000 |

## Mean rule penalty (before weighting)

| task | rule | sft | rl_best | rl_last |
|---|---|---|---|---|
| eng_hm | base_leak | 0.031 | 0.008 | 0.009 |
| eng_hm | base_presence | 0.003 | 0.003 | 0.003 |
| eng_hm | device_use | 0.079 | 0.028 | 0.039 |
| eng_hm | diacritic | 0.020 | 0.014 | 0.019 |
| eng_hm | draft | 0.000 | 0.000 | 0.000 |
| eng_hm | english | 0.000 | 0.000 | 0.000 |
| eng_hm | length | 0.005 | 0.006 | 0.005 |
| eng_hm | lid | 0.083 | 0.064 | 0.061 |
| eng_hm | lid_switch | 0.471 | 0.529 | 0.485 |
| eng_hm | loop | 0.002 | 0.000 | 0.000 |
| eng_hm | name | 0.056 | 0.044 | 0.039 |
| eng_hm | number | 0.001 | 0.000 | 0.000 |
| eng_hm | switch | 0.174 | 0.107 | 0.102 |
| eng_ood | base_leak | 0.000 | 0.000 | 0.000 |
| eng_ood | base_presence | 0.010 | 0.000 | 0.001 |
| eng_ood | device_use | 0.138 | 0.021 | 0.034 |
| eng_ood | diacritic | 0.031 | 0.002 | 0.006 |
| eng_ood | draft | 0.000 | 0.000 | 0.000 |
| eng_ood | english | 0.000 | 0.000 | 0.000 |
| eng_ood | length | 0.036 | 0.004 | 0.007 |
| eng_ood | lid | 0.068 | 0.030 | 0.022 |
| eng_ood | lid_switch | 0.000 | 0.000 | 0.000 |
| eng_ood | loop | 0.049 | 0.002 | 0.002 |
| eng_ood | name | 0.089 | 0.083 | 0.082 |
| eng_ood | number | 0.001 | 0.000 | 0.000 |
| eng_ood | switch | 0.000 | 0.000 | 0.000 |
| hm_ood | base_leak | 0.000 | 0.000 | 0.000 |
| hm_ood | base_presence | 0.008 | 0.000 | 0.002 |
| hm_ood | device_use | 0.146 | 0.027 | 0.035 |
| hm_ood | diacritic | 0.027 | 0.004 | 0.007 |
| hm_ood | draft | 0.000 | 0.000 | 0.000 |
| hm_ood | english | 0.000 | 0.000 | 0.000 |
| hm_ood | length | 0.033 | 0.003 | 0.004 |
| hm_ood | lid | 0.068 | 0.029 | 0.028 |
| hm_ood | lid_switch | 0.000 | 0.000 | 0.000 |
| hm_ood | loop | 0.055 | 0.002 | 0.002 |
| hm_ood | name | 0.000 | 0.000 | 0.000 |
| hm_ood | number | 0.001 | 0.000 | 0.000 |
| hm_ood | switch | 0.000 | 0.000 | 0.000 |
| m_hm | base_leak | 0.024 | 0.005 | 0.007 |
| m_hm | base_presence | 0.002 | 0.004 | 0.001 |
| m_hm | device_use | 0.085 | 0.019 | 0.031 |
| m_hm | diacritic | 0.010 | 0.005 | 0.007 |
| m_hm | draft | 0.000 | 0.000 | 0.000 |
| m_hm | english | 0.000 | 0.000 | 0.000 |
| m_hm | length | 0.001 | 0.001 | 0.001 |
| m_hm | lid | 0.088 | 0.083 | 0.075 |
| m_hm | lid_switch | 0.524 | 0.576 | 0.544 |
| m_hm | loop | 0.000 | 0.000 | 0.000 |
| m_hm | name | 0.000 | 0.000 | 0.000 |
| m_hm | number | 0.001 | 0.000 | 0.001 |
| m_hm | switch | 0.189 | 0.107 | 0.105 |
| m_ood | base_leak | 0.000 | 0.000 | 0.000 |
| m_ood | base_presence | 0.007 | 0.001 | 0.001 |
| m_ood | device_use | 0.137 | 0.021 | 0.029 |
| m_ood | diacritic | 0.028 | 0.004 | 0.004 |
| m_ood | draft | 0.000 | 0.000 | 0.000 |
| m_ood | english | 0.000 | 0.000 | 0.000 |
| m_ood | length | 0.023 | 0.004 | 0.002 |
| m_ood | lid | 0.069 | 0.029 | 0.025 |
| m_ood | lid_switch | 0.000 | 0.000 | 0.000 |
| m_ood | loop | 0.037 | 0.002 | 0.001 |
| m_ood | name | 0.000 | 0.000 | 0.000 |
| m_ood | number | 0.001 | 0.000 | 0.001 |
| m_ood | switch | 0.000 | 0.000 | 0.000 |

## Paired difference to sft (mean over items, 95% bootstrap CI)

| system | metric | all tasks | eng_hm | eng_ood | hm_ood | m_hm | m_ood |
|---|---|---|---|---|---|---|---|
| rl_best | evaluator | +0.227 [+0.215, +0.239] | +0.053 [+0.044, +0.063] | +0.329 [+0.299, +0.359] | +0.364 [+0.330, +0.401] | +0.061 [+0.050, +0.071] | +0.304 [+0.277, +0.333] |
| rl_best | reward | +0.211 [+0.202, +0.220] | +0.163 [+0.144, +0.181] | +0.244 [+0.223, +0.263] | +0.247 [+0.223, +0.271] | +0.171 [+0.154, +0.188] | +0.225 [+0.206, +0.245] |
| rl_last | evaluator | +0.207 [+0.195, +0.218] | +0.036 [+0.025, +0.046] | +0.306 [+0.277, +0.336] | +0.340 [+0.306, +0.375] | +0.055 [+0.043, +0.066] | +0.274 [+0.247, +0.302] |
| rl_last | reward | +0.205 [+0.196, +0.214] | +0.161 [+0.142, +0.179] | +0.235 [+0.214, +0.254] | +0.234 [+0.211, +0.258] | +0.174 [+0.156, +0.191] | +0.217 [+0.198, +0.238] |