#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/qparam_processing_cpu_benchmark_${RUN_ID}}"

MODELS="${MODELS:-olmoe}"
SCOPE="${SCOPE:-all_experts}"
GROUP_SIZES="${GROUP_SIZES:-1,16,32,64,128,256,512,1024,2048}"
VIRTUAL_REPEATS="${VIRTUAL_REPEATS:-1}"
QBITS="${QBITS:-4}"
WORK_ITERS="${WORK_ITERS:-16}"
WARMUP="${WARMUP:-5}"
TRIALS="${TRIALS:-20}"

mkdir -p "$OUT_ROOT"

python3 tools/qparam_processing_cpu_benchmark.py \
  --output_dir "$OUT_ROOT" \
  --models "$MODELS" \
  --scope "$SCOPE" \
  --group_sizes "$GROUP_SIZES" \
  --virtual_repeats "$VIRTUAL_REPEATS" \
  --qbits "$QBITS" \
  --work_iters "$WORK_ITERS" \
  --warmup "$WARMUP" \
  --trials "$TRIALS"

python3 tools/plot_qparam_processing_benchmark.py \
  --input_csv "$OUT_ROOT/summary_by_group.csv" \
  --output_dir "$OUT_ROOT/plots"

echo "[run-qparam-process-cpu] detail: $OUT_ROOT/detail.csv"
echo "[run-qparam-process-cpu] summary: $OUT_ROOT/summary_by_group.csv"
echo "[run-qparam-process-cpu] plots: $OUT_ROOT/plots"
