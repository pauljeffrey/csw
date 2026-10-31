#!/bin/bash
# Pilot: seeds + anchors + N generated samples + judge + adjudicate for every language, 4 languages at a time.
# usage: run_pilot.sh <python> <n> [langs...]
set -u
PY=$1; N=$2; shift 2
LANGS=${@:-"yor hau ibo pcm twi ewe fon kin sna zul xho swh orm som wol tsn nya afr sot"}
set -a; [ -f "${CSW_ENV_FILE:-.env}" ] && . "${CSW_ENV_FILE:-.env}"; set +a
one() {
  l=$1
  $PY -m csw.datagen.cs_pipeline seeds --langs $l
  [ -s anchors/$l.jsonl ] || $PY -m csw.datagen.cs_pipeline anchors --langs $l --per-cat 15 --conc 6
  $PY -m csw.datagen.cs_pipeline generate --langs $l --n $N --conc 6
  $PY -m csw.datagen.cs_pipeline judge --langs $l --conc 6
  $PY -m csw.datagen.cs_pipeline adjudicate --langs $l --conc 6
  $PY -m csw.datagen.cs_pipeline report --langs $l
}
export -f one; export PY N
echo $LANGS | tr ' ' '\n' | xargs -P 4 -I{} bash -c 'one {}' 2>&1
