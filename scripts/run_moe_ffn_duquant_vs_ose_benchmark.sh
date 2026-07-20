#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/moe_ffn_duquant_vs_ose_benchmark_${RUN_ID}}"

SCOPES="${SCOPES:-single_expert,moe_layer}"
PRESETS="${PRESETS:-olmoe}"
VARIANTS="${VARIANTS:-duquant_once,ours_ose,plain_no_ose,duquant_twice,component_duquant_transform,component_ose_branch}"
TOKENS_PER_EXPERT="${TOKENS_PER_EXPERT:-1,2,4,8,16,32}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-8,16,64}"
TOTAL_TOKENS="${TOTAL_TOKENS:-0}"
OSE_TOPK="${OSE_TOPK:-64}"
DUQUANT_BLOCK_SIZE="${DUQUANT_BLOCK_SIZE:-128}"
DUQUANT_ROTATION_STAGES="${DUQUANT_ROTATION_STAGES:-2}"
INCLUDE_DUQUANT_PERMUTATION="${INCLUDE_DUQUANT_PERMUTATION:-1}"
INCLUDE_SCATTER="${INCLUDE_SCATTER:-1}"
VALIDATE="${VALIDATE:-1}"

WARMUP="${WARMUP:-50}"
REPEAT="${REPEAT:-200}"
ROUNDS="${ROUNDS:-5}"
BLOCK_M="${BLOCK_M:-16}"
BLOCK_N="${BLOCK_N:-64}"
BLOCK_K="${BLOCK_K:-64}"
NUM_WARPS="${NUM_WARPS:-4}"
ELEM_BLOCK="${ELEM_BLOCK:-1024}"
SEED="${SEED:-2}"

mkdir -p "$OUT_ROOT"

duquant_perm_arg=()
if [[ "$INCLUDE_DUQUANT_PERMUTATION" == "0" || "$INCLUDE_DUQUANT_PERMUTATION" == "false" || "$INCLUDE_DUQUANT_PERMUTATION" == "FALSE" ]]; then
  duquant_perm_arg=(--no_include_duquant_permutation)
fi

scatter_arg=()
if [[ "$INCLUDE_SCATTER" == "0" || "$INCLUDE_SCATTER" == "false" || "$INCLUDE_SCATTER" == "FALSE" ]]; then
  scatter_arg=(--no_include_scatter)
fi

validate_arg=()
if [[ "$VALIDATE" == "0" || "$VALIDATE" == "false" || "$VALIDATE" == "FALSE" ]]; then
  validate_arg=(--no_validate)
fi

echo "[run-moe-ffn-duquant-vs-ose] output: $OUT_ROOT"
echo "[run-moe-ffn-duquant-vs-ose] scopes: $SCOPES"
echo "[run-moe-ffn-duquant-vs-ose] presets: $PRESETS"
echo "[run-moe-ffn-duquant-vs-ose] variants: $VARIANTS"
echo "[run-moe-ffn-duquant-vs-ose] tokens_per_expert: $TOKENS_PER_EXPERT"
echo "[run-moe-ffn-duquant-vs-ose] active_experts: $ACTIVE_EXPERTS"
echo "[run-moe-ffn-duquant-vs-ose] warmup=$WARMUP repeat=$REPEAT rounds=$ROUNDS"

python3 tools/triton_moe_ffn_duquant_vs_ose_benchmark.py \
  --output_dir "$OUT_ROOT" \
  --scopes "$SCOPES" \
  --presets "$PRESETS" \
  --variants "$VARIANTS" \
  --tokens_per_expert "$TOKENS_PER_EXPERT" \
  --active_experts "$ACTIVE_EXPERTS" \
  --total_tokens "$TOTAL_TOKENS" \
  --ose_topk "$OSE_TOPK" \
  --duquant_block_size "$DUQUANT_BLOCK_SIZE" \
  --duquant_rotation_stages "$DUQUANT_ROTATION_STAGES" \
  "${duquant_perm_arg[@]}" \
  "${scatter_arg[@]}" \
  "${validate_arg[@]}" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --rounds "$ROUNDS" \
  --block_m "$BLOCK_M" \
  --block_n "$BLOCK_N" \
  --block_k "$BLOCK_K" \
  --num_warps "$NUM_WARPS" \
  --elem_block "$ELEM_BLOCK" \
  --seed "$SEED"

python3 tools/plot_moe_ffn_duquant_vs_ose_benchmark.py \
  --input_csv "$OUT_ROOT/moe_ffn_duquant_vs_ose_detail.csv" \
  --output_dir "$OUT_ROOT/plots"

echo "[run-moe-ffn-duquant-vs-ose] done"
echo "[run-moe-ffn-duquant-vs-ose] detail: $OUT_ROOT/moe_ffn_duquant_vs_ose_detail.csv"
echo "[run-moe-ffn-duquant-vs-ose] report: $OUT_ROOT/report.md"
echo "[run-moe-ffn-duquant-vs-ose] plots: $OUT_ROOT/plots"
