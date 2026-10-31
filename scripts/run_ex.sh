#!/bin/bash
# Exemplar source rows: DeepSeek (thinking off) zero-shot -> Qwen judge -> Kimi adjudicate; then top rows become exemplars.
PY=$1; N=$2; shift 2; LANGS=${@:-"yor zul swh pcm"}
set -a; [ -f "${CSW_ENV_FILE:-.env}" ] && . "${CSW_ENV_FILE:-.env}"; set +a
export CS_OUT=out_ex CS_SEED=100 CS_REASONING=none CS_ADJ_TOK=3000
one() { l=$1; C="--provider doubleword --conc 4 --langs $l"
  $PY -m csw.datagen.cs_pipeline generate $C --model deepseek-ai/DeepSeek-V4-Pro --n $N
  $PY -m csw.datagen.cs_pipeline judge $C --judge-model Qwen/Qwen3.5-397B-A17B-FP8
  $PY -m csw.datagen.cs_pipeline adjudicate $C --adj-model moonshotai/kimi-k3 --thr 0.5
  $PY -m csw.datagen.exemplars $l out_ex 5 0.5; }
export -f one; export PY N
echo $LANGS | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'one {}' 2>&1
