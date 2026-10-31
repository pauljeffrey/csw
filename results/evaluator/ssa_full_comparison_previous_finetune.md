# Full SSA-MTE test (16 languages) + AfriMTE eng->X: 20204 human-rated segments

Spearman pooled / with per-language-pair means removed (the second removes between-language offsets). 95% bootstrap CI for the pooled value.

| scorer | vs human z | CI | within-pair |
|---|---|---|---|
| our evaluator cft2b|strict, mean M1-M6 | 0.278 | [0.264, 0.290] | 0.344 |
| our evaluator cft2b|std, mean M1-M6 | 0.289 | [0.276, 0.302] | 0.340 |
| our evaluator v3|-, mean M1-M6 | 0.270 | [0.257, 0.282] | 0.327 |
| SSA-COMET-MTL (ref) | 0.575 | [0.566, 0.585] | 0.574 |
| SSA-COMET-MTL (QE) | 0.536 | [0.526, 0.546] | 0.530 |
| SSA-COMET-QE | 0.491 | [0.481, 0.503] | 0.482 |
| AfriCOMET-STL1.1 (ref) | 0.430 | [0.419, 0.443] | 0.478 |
| AfriCOMET-MTL (ref) | 0.331 | [0.319, 0.344] | 0.446 |
| AfriCOMET-QE-STL1.1 | 0.348 | [0.336, 0.363] | 0.402 |

## Single dimensions of our evaluator (cft2b|strict) vs human z and the metrics

| dim | human z | SSA-COMET-MTL (ref) | SSA-COMET-MTL (QE) | SSA-COMET-QE | AfriCOMET-STL1.1 (ref) | AfriCOMET-MTL (ref) | AfriCOMET-QE-STL1.1 |
|---|---|---|---|---|---|---|---|
| M1 faithfulness | 0.269 | 0.392 | 0.387 | 0.413 | 0.506 | 0.563 | 0.477 |
| M2 fluency | 0.284 | 0.402 | 0.392 | 0.420 | 0.507 | 0.557 | 0.474 |
| M3 orthography | 0.272 | 0.399 | 0.382 | 0.398 | 0.477 | 0.551 | 0.466 |
| M4 language identity | 0.144 | 0.179 | 0.155 | 0.186 | 0.361 | 0.375 | 0.267 |
| M5 register | 0.275 | 0.383 | 0.370 | 0.396 | 0.513 | 0.566 | 0.455 |
| M6 terminology | 0.269 | 0.394 | 0.385 | 0.417 | 0.524 | 0.579 | 0.484 |

## Regression of each target on M1..M6 (standardised; cross-validated leaving one language pair out)

| target | R2 in-sample | R2 CV | CV Spearman | mean-M Spearman | NNLS shares % (M1..M6) |
|---|---|---|---|---|---|
| human z | 0.131 | 0.067 | 0.191 | 0.278 | 0, 0, 0, 0, 100, 0 |
| SSA-COMET-MTL (ref) | 0.304 | 0.265 | 0.357 | 0.395 | 0, 0, 36, 0, 64, 0 |
| SSA-COMET-MTL (QE) | 0.301 | 0.259 | 0.340 | 0.382 | 3, 0, 28, 0, 69, 0 |
| SSA-COMET-QE | 0.320 | 0.267 | 0.357 | 0.410 | 4, 0, 26, 0, 70, 0 |
| AfriCOMET-STL1.1 (ref) | 0.389 | 0.314 | 0.461 | 0.525 | 14, 0, 0, 7, 79, 0 |
| AfriCOMET-MTL (ref) | 0.464 | 0.387 | 0.517 | 0.582 | 20, 0, 12, 3, 65, 0 |
| AfriCOMET-QE-STL1.1 | 0.352 | 0.265 | 0.412 | 0.479 | 21, 0, 34, 0, 44, 0 |

## Per language pair: Spearman with human z

| pair | n | our mean-M | SSA-COMET-MTL (ref) | SSA-COMET-MTL (QE) | SSA-COMET-QE | AfriCOMET-STL1.1 (ref) | AfriCOMET-MTL (ref) | AfriCOMET-QE-STL1.1 |
|---|---|---|---|---|---|---|---|---|
| eng-amh | 1166 | 0.201 | 0.630 | 0.594 | 0.552 | 0.588 | 0.613 | 0.547 |
| eng-ewe | 1077 | 0.345 | 0.639 | 0.591 | 0.526 | 0.484 | 0.473 | 0.388 |
| eng-hau | 1432 | 0.192 | 0.502 | 0.436 | 0.390 | 0.461 | 0.477 | 0.354 |
| eng-ibo | 1008 | 0.187 | 0.352 | 0.314 | 0.282 | 0.289 | 0.344 | 0.260 |
| eng-kik | 1374 | 0.441 | 0.722 | 0.693 | 0.657 | 0.507 | 0.539 | 0.586 |
| eng-kin | 1210 | 0.364 | 0.601 | 0.563 | 0.511 | 0.507 | 0.546 | 0.397 |
| eng-lin | 1185 | 0.259 | 0.512 | 0.479 | 0.373 | 0.349 | 0.341 | 0.305 |
| eng-luo | 1441 | 0.336 | 0.737 | 0.693 | 0.618 | 0.593 | 0.459 | 0.546 |
| eng-nya | 1242 | 0.612 | 0.695 | 0.670 | 0.670 | 0.685 | 0.679 | 0.665 |
| eng-som | 226 | 0.377 | 0.519 | 0.456 | 0.460 | 0.502 | 0.543 | 0.458 |
| eng-sot | 848 | 0.226 | 0.448 | 0.406 | 0.427 | 0.304 | 0.381 | 0.271 |
| eng-swh | 1660 | 0.387 | 0.297 | 0.217 | 0.196 | 0.458 | 0.282 | 0.147 |
| eng-twi | 1447 | 0.339 | 0.622 | 0.593 | 0.565 | 0.483 | 0.466 | 0.426 |
| eng-vmw | 930 | 0.272 | 0.328 | 0.249 | 0.199 | 0.239 | 0.280 | 0.120 |
| eng-wol | 1175 | 0.306 | 0.730 | 0.690 | 0.638 | 0.493 | 0.471 | 0.434 |
| eng-xho | 243 | 0.100 | 0.154 | 0.150 | 0.146 | 0.203 | 0.138 | 0.117 |
| eng-yor | 1445 | 0.377 | 0.529 | 0.505 | 0.445 | 0.436 | 0.447 | 0.378 |
| eng-zul | 1095 | 0.139 | 0.282 | 0.223 | 0.232 | 0.266 | 0.290 | 0.209 |