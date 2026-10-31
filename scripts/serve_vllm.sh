#!/usr/bin/env bash
# Serve Gemma-4-31B for generation. Highest precision that fits (low-resource languages are what suffers from quantisation):
#   >= 90 GB VRAM (RTX PRO 6000 96 GB) -> google/gemma-4-31B-it in bf16
#   smaller (A100/H100 80 GB, 68 GB disk) -> RedHatAI/gemma-4-31B-it-FP8-dynamic (compressed-tensors FP8, ~33 GB; near-lossless; Marlin weight-only on A100)
# The served name stays google/gemma-4-31b-it so the pipeline's --model flag never changes. Prefix caching is ON (the ~8k-token system message is shared per language).
set -euo pipefail
PORT=${PORT:-8000}; NAME=${NAME:-google/gemma-4-31b-it}
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${VRAM:-0}" -ge 90000 ]; then REPO=${REPO:-google/gemma-4-31B-it}; else REPO=${REPO:-RedHatAI/gemma-4-31B-it-FP8-dynamic}; fi
echo "serving $REPO as $NAME ($VRAM MiB)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
NV=$(python3 -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13','lib'))" 2>/dev/null || true)   # CUDA-13 torch wheels: nvrtc builtins live inside the venv
[ -d "$NV" ] && export LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}"
EXTRA=""
[ -n "${KV_DTYPE:-}" ] && EXTRA="$EXTRA --kv-cache-dtype $KV_DTYPE"                                   # fp8 KV halves KV memory -> about twice the concurrent requests
[ -n "${KV_OFFLOAD_GB:-}" ] && EXTRA="$EXTRA --kv-offloading-backend native --kv-offloading-size $KV_OFFLOAD_GB"   # spill cold KV / prefix blocks to host RAM (the boxes have 500 GB)
exec vllm serve $REPO --served-model-name $NAME --max-model-len ${MAXLEN:-32768} --gpu-memory-utilization ${GPU_UTIL:-0.92} \
  --enable-prefix-caching --max-num-seqs ${SEQS:-48} ${BATCHED_TOKENS:+--max-num-batched-tokens $BATCHED_TOKENS} --port $PORT $EXTRA
