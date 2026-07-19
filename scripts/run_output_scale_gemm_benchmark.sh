#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/output_scale_gemm_benchmark_${RUN_ID}}"

SCOPES="${SCOPES:-full_gemm,epilogue_only,grouped_moe}"
SHAPES="${SHAPES:-olmoe_gate_up,olmoe_down}"
M_VALUES="${M_VALUES:-1,2,4,8,16,32,64,128}"
NUM_ACTIVE_EXPERTS="${NUM_ACTIVE_EXPERTS:-2,4,8,16}"
TOKENS_PER_EXPERT="${TOKENS_PER_EXPERT:-1,2,4,8}"
GRANULARITIES="${GRANULARITIES:-per_channel,2,4,8,16,per_tensor}"

WARMUP="${WARMUP:-200}"
REPEAT="${REPEAT:-1000}"
ROUNDS="${ROUNDS:-5}"
BLOCK_M="${BLOCK_M:-16}"
BLOCK_N="${BLOCK_N:-64}"
BLOCK_K="${BLOCK_K:-64}"
NUM_WARPS="${NUM_WARPS:-4}"

mkdir -p "$OUT_ROOT"

echo "[run-output-scale] output: $OUT_ROOT"
echo "[run-output-scale] scopes: $SCOPES"
echo "[run-output-scale] shapes: $SHAPES"
echo "[run-output-scale] m_values: $M_VALUES"
echo "[run-output-scale] active_experts: $NUM_ACTIVE_EXPERTS"
echo "[run-output-scale] tokens_per_expert: $TOKENS_PER_EXPERT"
echo "[run-output-scale] granularities: $GRANULARITIES"
echo "[run-output-scale] warmup=$WARMUP repeat=$REPEAT rounds=$ROUNDS"

python3 tools/triton_output_scale_gemm_benchmark.py \
  --output_dir "$OUT_ROOT" \
  --scopes "$SCOPES" \
  --shapes "$SHAPES" \
  --m_values "$M_VALUES" \
  --num_active_experts "$NUM_ACTIVE_EXPERTS" \
  --tokens_per_expert "$TOKENS_PER_EXPERT" \
  --granularities "$GRANULARITIES" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --rounds "$ROUNDS" \
  --block_m "$BLOCK_M" \
  --block_n "$BLOCK_N" \
  --block_k "$BLOCK_K" \
  --num_warps "$NUM_WARPS"

python3 tools/plot_output_scale_gemm_benchmark.py \
  --input_csv "$OUT_ROOT/output_scale_gemm_detail.csv" \
  --metadata_csv "$OUT_ROOT/olmoe_scale_metadata_summary.csv" \
  --output_dir "$OUT_ROOT/plots"

echo "[run-output-scale] done"
echo "[run-output-scale] detail: $OUT_ROOT/output_scale_gemm_detail.csv"
echo "[run-output-scale] metadata: $OUT_ROOT/olmoe_scale_metadata_summary.csv"
echo "[run-output-scale] plots: $OUT_ROOT/plots"
