#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/group_scale_triton_benchmark_${RUN_ID}}"

SHAPES="${SHAPES:-olmoe_gate_up,olmoe_down}"
M_VALUES="${M_VALUES:-1,4,8,16,32,64}"
GROUP_SIZES="${GROUP_SIZES:-1,16,32,64,128,256,512,1024,2048}"
WARMUP="${WARMUP:-50}"
REPEAT="${REPEAT:-200}"
TRIALS="${TRIALS:-5}"
BLOCK_M="${BLOCK_M:-16}"
BLOCK_N="${BLOCK_N:-64}"
BLOCK_K="${BLOCK_K:-64}"
NUM_WARPS="${NUM_WARPS:-4}"

mkdir -p "$OUT_ROOT"

echo "[run-group-scale-bench] output: $OUT_ROOT"
echo "[run-group-scale-bench] shapes: $SHAPES"
echo "[run-group-scale-bench] m_values: $M_VALUES"
echo "[run-group-scale-bench] group_sizes: $GROUP_SIZES"
echo "[run-group-scale-bench] warmup=$WARMUP repeat=$REPEAT trials=$TRIALS"

python3 tools/triton_group_scale_benchmark.py \
  --output_dir "$OUT_ROOT" \
  --shapes "$SHAPES" \
  --m_values "$M_VALUES" \
  --group_sizes "$GROUP_SIZES" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --trials "$TRIALS" \
  --block_m "$BLOCK_M" \
  --block_n "$BLOCK_N" \
  --block_k "$BLOCK_K" \
  --num_warps "$NUM_WARPS"

echo "[run-group-scale-bench] done"
echo "[run-group-scale-bench] detail: $OUT_ROOT/detail.csv"
echo "[run-group-scale-bench] summary: $OUT_ROOT/summary_by_group.csv"
