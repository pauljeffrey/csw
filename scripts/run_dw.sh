#!/bin/bash
# Staged run on Doubleword: generate=Gemma-4-31B (v2 prompt, thinking off), judge=Qwen3.5-397B, adjudicate=Kimi-K3.
# anchors: DeepSeek-V4-Pro (thinking off, one-time). usage: run_dw.sh <python> <n> [langs...]
set -u
PY=$1; N=$2; shift 2
LANGS=${@:-"yor pcm swh zul hau ibo twi ewe fon kin sna xho orm som wol tsn nya afr sot efi ibb urh"}
set -a; [ -f "${CSW_ENV_FILE:-.env}" ] && . "${CSW_ENV_FILE:-.env}"; set +a
export CS_OUT=out_dw2 CS_ADJ_TOK=3000 CS_REASONING=none CS_PROMPT=v2 CS_SEED=0
one() {
  l=$1; C="--provider doubleword --conc 4 --langs $l"
  $PY -m csw.datagen.cs_pipeline seeds --langs $l >/dev/null 2>&1
  $PY -m csw.datagen.cs_pipeline anchors $C --model deepseek-ai/DeepSeek-V4-Pro --per-cat 20
  $PY -m csw.datagen.cs_pipeline generate $C --model google/gemma-4-31B-it --n $N
  $PY -m csw.datagen.cs_pipeline judge $C --judge-model Qwen/Qwen3.5-397B-A17B-FP8
  $PY -m csw.datagen.cs_pipeline adjudicate $C --adj-model moonshotai/kimi-k3 --thr 0.5
  $PY -m csw.datagen.cs_pipeline report $C
}
export -f one; export PY N
echo $LANGS | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'one {}' 2>&1
