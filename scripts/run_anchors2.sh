#!/bin/bash
# anchors for the 18 added languages on OpenRouter (DeepSeek-V4-Pro, reasoning off): round 1 then round 2.
PY=$1; shift; set -a; [ -f "${CSW_ENV_FILE:-.env}" ] && . "${CSW_ENV_FILE:-.env}"; set +a
export CS_REASONING=none
one() { l=$1
  $PY -m csw.datagen.cs_pipeline anchors --provider openrouter --conc 4 --langs $l --model deepseek/deepseek-v4-pro --per-cat 25 --round 1
  $PY -m csw.datagen.cs_pipeline anchors --provider openrouter --conc 4 --langs $l --model deepseek/deepseek-v4-pro --per-cat 30 --round 2; }
export -f one; export PY
echo "tum tso nso lug kik luo mos ssw nde umb kmb kon sag ful mlg amh tir bin" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'one {}' 2>&1
