#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/int8_quant_linear_group_benchmark_${RUN_ID}}"

SCOPES="${SCOPES:-single_gemm,epilogue_only,moe_expert,moe_epilogue_only}"
LINEAR_SHAPES="${LINEAR_SHAPES:-olmoe_gate,olmoe_down}"
M_VALUES="${M_VALUES:-1,2,4,8,16,32}"
MOE_PRESETS="${MOE_PRESETS:-olmoe}"
NUM_ACTIVE_EXPERTS="${NUM_ACTIVE_EXPERTS:-2,4,8}"
TOKENS_PER_EXPERT="${TOKENS_PER_EXPERT:-1,2,4,8}"
GROUP_SIZES="${GROUP_SIZES:-1,2,4,8,16,32,64,128,256,512,1024,2048}"
ACT_SCALE_MODES="${ACT_SCALE_MODES:-per_token,per_tensor_static}"

WARMUP="${WARMUP:-100}"
REPEAT="${REPEAT:-500}"
ROUNDS="${ROUNDS:-5}"
BLOCK_M="${BLOCK_M:-16}"
BLOCK_N="${BLOCK_N:-64}"
BLOCK_K="${BLOCK_K:-64}"
NUM_WARPS="${NUM_WARPS:-4}"

mkdir -p "$OUT_ROOT"

echo "[run-int8-group-bench] output: $OUT_ROOT"
echo "[run-int8-group-bench] scopes: $SCOPES"
echo "[run-int8-group-bench] linear_shapes: $LINEAR_SHAPES"
echo "[run-int8-group-bench] m_values: $M_VALUES"
echo "[run-int8-group-bench] moe_presets: $MOE_PRESETS"
echo "[run-int8-group-bench] active_experts: $NUM_ACTIVE_EXPERTS"
echo "[run-int8-group-bench] tokens_per_expert: $TOKENS_PER_EXPERT"
echo "[run-int8-group-bench] group_sizes: $GROUP_SIZES"
echo "[run-int8-group-bench] act_scale_modes: $ACT_SCALE_MODES"
echo "[run-int8-group-bench] warmup=$WARMUP repeat=$REPEAT rounds=$ROUNDS"

python3 tools/triton_int8_quant_linear_group_benchmark.py \
  --output_dir "$OUT_ROOT" \
  --scopes "$SCOPES" \
  --linear_shapes "$LINEAR_SHAPES" \
  --m_values "$M_VALUES" \
  --moe_presets "$MOE_PRESETS" \
  --num_active_experts "$NUM_ACTIVE_EXPERTS" \
  --tokens_per_expert "$TOKENS_PER_EXPERT" \
  --group_sizes "$GROUP_SIZES" \
  --act_scale_modes "$ACT_SCALE_MODES" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --rounds "$ROUNDS" \
  --block_m "$BLOCK_M" \
  --block_n "$BLOCK_N" \
  --block_k "$BLOCK_K" \
  --num_warps "$NUM_WARPS"

python3 tools/plot_int8_quant_linear_group_benchmark.py \
  --input_csv "$OUT_ROOT/int8_quant_linear_group_detail.csv" \
  --output_dir "$OUT_ROOT/plots"

echo "[run-int8-group-bench] done"
echo "[run-int8-group-bench] detail: $OUT_ROOT/int8_quant_linear_group_detail.csv"
echo "[run-int8-group-bench] report: $OUT_ROOT/report.md"
echo "[run-int8-group-bench] plots: $OUT_ROOT/plots"
