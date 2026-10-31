#!/usr/bin/env bash
# Generation on ANY GPU box once a vLLM server is up (OpenAI-compatible on :8000). Per-language targets come from profiles.TARGETS.
#   MODEL=google/gemma-4-31b-it CONC=64 CS_OUT=out_main bash scripts/gen_rounds.sh
# Rounds: fractions of every language's target; ONE shared worker pool so languages advance together (never one-language-at-a-time).
# Pidgin (LAST) runs after everything else because track D reuses the English already generated for yor/hau/ibo/twi/ewe/zul.
set -uo pipefail
export CS_OUT=${CS_OUT:-out_main} CS_SEED=0 CS_LOCAL_URL=${CS_LOCAL_URL:-http://127.0.0.1:8000/v1}
PY=${PY:-python3}; MODEL=${MODEL:-google/gemma-4-31b-it}; FRACS=${FRACS:-"0.02 0.08 0.25 0.6 1.0"}; CONC=${CONC:-48}
export CS_TEMP=${CS_TEMP:-0.6} CS_REP_PEN=${CS_REP_PEN:-1.05}     # sampling: temperature 0.6, top-p 0.95, top-k 40, repetition penalty 1.05 (see cs_pipeline.prov)
# CS_THINK=1 asks the chat template for thinking mode - untested on Gemma-4; try on 20 rows and compare rule_score before using it
# Sampling is chosen EMPIRICALLY once per run: 60 identical task cards x 4 configs, scored by the rule-based scorer (csw.datagen.pilot_sampling -> gen_config.json)
if [ ! -f gen_config.json ] && [ "${PILOT:-1}" = 1 ]; then $PY -m csw.datagen.pilot_sampling --model $MODEL; fi
if [ -f gen_config.json ]; then
  eval $($PY -c "import json;c=json.load(open('gen_config.json'))['chosen'];print('export CS_TEMP=%s CS_REP_PEN=%s CS_TOP_P=%s'%(c['temperature'],c['rep'],c['top_p']))")
fi
echo "sampling: temp=$CS_TEMP rep_pen=$CS_REP_PEN top_p=${CS_TOP_P:-0.95}"
MAIN=$($PY -c "from csw.datagen.profiles import TARGETS,LAST,HOLD;print(' '.join(l for l in TARGETS if l not in LAST and l not in HOLD))")
HELD=$($PY -c "from csw.datagen.profiles import HOLD;print(' '.join(HOLD))")   # efi ibb urh bin mos kmb umb sag: Gemma cannot write them yet
LAST=$($PY -c "from csw.datagen.profiles import LAST;print(' '.join(LAST))")
# Order (compute is the limit, so the strongest languages come first; 80-95% quality is acceptable, even if only for high-resource languages):
#   1. OK-tier languages (profiles.SUPPORT["ok"]) at fractions A   2. English pool -> Pidgins (pcm gpe wes) at fractions A   3. OK-tier + Pidgins at fractions B
#   4. borderline-tier languages (all fractions)   5. held languages only with RUN_HELD=1
OKL=$($PY -c "from csw.datagen.profiles import TARGETS,LAST,HOLD,SUPPORT;print(' '.join(l for l in TARGETS if l in SUPPORT['ok'] and l not in LAST and l not in HOLD))")
BORDER=$($PY -c "from csw.datagen.profiles import TARGETS,LAST,HOLD,SUPPORT;print(' '.join(l for l in TARGETS if l not in SUPPORT['ok'] and l not in LAST and l not in HOLD))")
FRACS_A=${FRACS_A:-"0.02 0.08 0.25"}; FRACS_B=${FRACS_B:-"0.6 1.0"}
run() { for f in $2; do $PY -m csw.datagen.cs_pipeline genall --provider local --model $MODEL --langs $1 --n 0 --frac $f --conc $CONC; echo "ROUND_DONE [$3] frac=$f $(date -u +%H:%M)"; done; }
run "$OKL" "$FRACS_A" ok-A
$PY -m csw.datagen.build_eng_pool $CS_OUT          # English from the finished rows -> given English (track D) for Pidgin and the small languages
run "$LAST" "$FRACS_A" pidgin-A
run "$OKL" "$FRACS_B" ok-B
$PY -m csw.datagen.build_eng_pool $CS_OUT
run "$LAST" "$FRACS_B" pidgin-B
run "$BORDER" "$FRACS_A $FRACS_B" borderline
if [ "${RUN_HELD:-0}" = 1 ]; then      # only when time/compute remain (RUN_HELD=1)
  for f in $FRACS; do
    $PY -m csw.datagen.cs_pipeline genall --provider local --model $MODEL --langs $HELD --n 0 --frac $f --conc $CONC
    echo "ROUND_DONE held frac=$f $(date -u +%H:%M)"
  done
fi
echo ALL_DONE
