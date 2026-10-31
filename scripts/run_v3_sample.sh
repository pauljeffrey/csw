#!/bin/bash
# v3 sample on OpenRouter : Gemma-4-31B generates, DeepSeek-V4-Pro (reasoning off) judges.
PY=$1; LIM=${2:-3}; set -a; [ -f "${CSW_ENV_FILE:-.env}" ] && . "${CSW_ENV_FILE:-.env}"; set +a
export CS_OUT=out_v3b CS_SEED=0 CS_REASONING=none
LANGS=$($PY -c "from csw.datagen import profiles;print(' '.join(profiles.LANGS))")
one() { l=$1; C="--provider openrouter --conc 4 --langs $l"
  $PY -m csw.datagen.cs_pipeline generate $C --model google/gemma-4-31b-it --n 6000 --limit $LIM
  $PY -m csw.datagen.cs_pipeline judge $C --judge-model deepseek/deepseek-v4-pro; }
export -f one; export PY LIM
echo $LANGS | tr ' ' '\n' | xargs -P 4 -I{} bash -c 'one {}' 2>&1
