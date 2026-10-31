#!/bin/bash
# anchors: round 1 for new languages, round 2 (different, rarer items) for all. DeepSeek-V4-Pro, thinking off.
PY=$1; shift; set -a; [ -f "${CSW_ENV_FILE:-.env}" ] && . "${CSW_ENV_FILE:-.env}"; set +a
export CS_REASONING=none
one() { l=$1
  $PY -m csw.datagen.cs_pipeline anchors --provider doubleword --conc 4 --langs $l --model deepseek-ai/DeepSeek-V4-Pro --per-cat 25 --round 1
  $PY -m csw.datagen.cs_pipeline anchors --provider doubleword --conc 4 --langs $l --model deepseek-ai/DeepSeek-V4-Pro --per-cat 30 --round 2; }
export -f one; export PY
echo "yor hau ibo pcm twi ewe fon kin sna zul xho swh orm som wol tsn nya afr sot efi ibb urh lin run bam" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'one {}' 2>&1
