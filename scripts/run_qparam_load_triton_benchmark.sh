#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/qparam_load_triton_benchmark_${RUN_ID}}"

MODELS="${MODELS:-olmoe}"
SCOPE="${SCOPE:-all_experts}"
GROUP_SIZES="${GROUP_SIZES:-1,16,32,64,128,256,512,1024,2048}"
VIRTUAL_REPEATS="${VIRTUAL_REPEATS:-1}"
WARMUP="${WARMUP:-50}"
REPEAT="${REPEAT:-200}"
TRIALS="${TRIALS:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
QPARAM_BYTES_PER_GROUP="${QPARAM_BYTES_PER_GROUP:-4}"

mkdir -p "$OUT_ROOT"

echo "[run-qparam-load-bench] output: $OUT_ROOT"
echo "[run-qparam-load-bench] models: $MODELS"
echo "[run-qparam-load-bench] scope: $SCOPE"
echo "[run-qparam-load-bench] group_sizes: $GROUP_SIZES"
echo "[run-qparam-load-bench] virtual_repeats: $VIRTUAL_REPEATS"
echo "[run-qparam-load-bench] warmup=$WARMUP repeat=$REPEAT trials=$TRIALS"

python3 tools/triton_qparam_load_benchmark.py \
  --output_dir "$OUT_ROOT" \
  --models "$MODELS" \
  --scope "$SCOPE" \
  --group_sizes "$GROUP_SIZES" \
  --virtual_repeats "$VIRTUAL_REPEATS" \
  --qparam_bytes_per_group "$QPARAM_BYTES_PER_GROUP" \
  --block_size "$BLOCK_SIZE" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --trials "$TRIALS"

echo "[run-qparam-load-bench] done"
echo "[run-qparam-load-bench] detail: $OUT_ROOT/detail.csv"
echo "[run-qparam-load-bench] summary: $OUT_ROOT/summary_by_group.csv"
