#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/qparam_processing_triton_benchmark_${RUN_ID}}"

MODELS="${MODELS:-olmoe}"
SCOPE="${SCOPE:-all_experts}"
GROUP_SIZES="${GROUP_SIZES:-1,16,32,64,128,256,512,1024,2048}"
VIRTUAL_REPEATS="${VIRTUAL_REPEATS:-1}"
QBITS="${QBITS:-4}"
WORK_ITERS="${WORK_ITERS:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
WARMUP="${WARMUP:-200}"
REPEAT="${REPEAT:-1000}"
TRIALS="${TRIALS:-5}"

mkdir -p "$OUT_ROOT"

echo "[run-qparam-process] output: $OUT_ROOT"
echo "[run-qparam-process] models: $MODELS"
echo "[run-qparam-process] scope: $SCOPE"
echo "[run-qparam-process] group_sizes: $GROUP_SIZES"
echo "[run-qparam-process] virtual_repeats: $VIRTUAL_REPEATS"
echo "[run-qparam-process] qbits=$QBITS work_iters=$WORK_ITERS block_size=$BLOCK_SIZE warmup=$WARMUP repeat=$REPEAT trials=$TRIALS"

python3 tools/triton_qparam_processing_benchmark.py \
  --output_dir "$OUT_ROOT" \
  --models "$MODELS" \
  --scope "$SCOPE" \
  --group_sizes "$GROUP_SIZES" \
  --virtual_repeats "$VIRTUAL_REPEATS" \
  --qbits "$QBITS" \
  --work_iters "$WORK_ITERS" \
  --block_size "$BLOCK_SIZE" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --trials "$TRIALS"

python3 tools/plot_qparam_processing_benchmark.py \
  --input_csv "$OUT_ROOT/summary_by_group.csv" \
  --output_dir "$OUT_ROOT/plots"

echo "[run-qparam-process] done"
echo "[run-qparam-process] detail: $OUT_ROOT/detail.csv"
echo "[run-qparam-process] summary: $OUT_ROOT/summary_by_group.csv"
echo "[run-qparam-process] plots: $OUT_ROOT/plots"
