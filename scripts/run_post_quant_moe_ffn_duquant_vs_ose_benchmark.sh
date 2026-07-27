#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/post_quant_moe_ffn_duquant_vs_ose_${RUN_ID}}"

PRESETS="${PRESETS:-olmoe}"
SCOPE="${SCOPE:-expert_ffn}"
VARIANTS="${VARIANTS:-duquant_once,plain_no_ose,ours_ose_w8a8,component_gateup_duquant_transform,component_ose_w8a8_branch}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-1,8,16,64}"
TOKENS_PER_EXPERT="${TOKENS_PER_EXPERT:-1,2,4,8,16,32}"
TOTAL_TOKENS="${TOTAL_TOKENS:-0}"
OSE_TOPK="${OSE_TOPK:-64}"

MAIN_WEIGHT_GROUP_SIZE="${MAIN_WEIGHT_GROUP_SIZE:-2048}"
OSE_WEIGHT_GROUP_SIZE="${OSE_WEIGHT_GROUP_SIZE:-1}"
DUQUANT_BLOCK_SIZE="${DUQUANT_BLOCK_SIZE:-128}"
DUQUANT_ROTATION_STAGES="${DUQUANT_ROTATION_STAGES:-2}"
INCLUDE_DUQUANT_PERMUTATION="${INCLUDE_DUQUANT_PERMUTATION:-1}"
INCLUDE_DOWN_DUQUANT_TRANSFORM="${INCLUDE_DOWN_DUQUANT_TRANSFORM:-1}"

WARMUP="${WARMUP:-30}"
REPEAT="${REPEAT:-100}"
ROUNDS="${ROUNDS:-5}"
VALIDATE="${VALIDATE:-0}"
BLOCK_M="${BLOCK_M:-16}"
BLOCK_N="${BLOCK_N:-64}"
BLOCK_K="${BLOCK_K:-64}"
NUM_WARPS="${NUM_WARPS:-4}"
ELEM_BLOCK="${ELEM_BLOCK:-1024}"
SEED="${SEED:-3}"

mkdir -p "$OUT_ROOT"

perm_arg=()
if [[ "$INCLUDE_DUQUANT_PERMUTATION" == "0" || "$INCLUDE_DUQUANT_PERMUTATION" == "false" || "$INCLUDE_DUQUANT_PERMUTATION" == "FALSE" ]]; then
  perm_arg=(--no_include_duquant_permutation)
fi

down_arg=()
if [[ "$INCLUDE_DOWN_DUQUANT_TRANSFORM" == "0" || "$INCLUDE_DOWN_DUQUANT_TRANSFORM" == "false" || "$INCLUDE_DOWN_DUQUANT_TRANSFORM" == "FALSE" ]]; then
  down_arg=(--no_include_down_duquant_transform)
fi

validate_arg=()
if [[ "$VALIDATE" == "0" || "$VALIDATE" == "false" || "$VALIDATE" == "FALSE" ]]; then
  validate_arg=(--no_validate)
fi

echo "[run-post-quant-ffn] output: $OUT_ROOT"
echo "[run-post-quant-ffn] presets: $PRESETS"
echo "[run-post-quant-ffn] scope: $SCOPE"
echo "[run-post-quant-ffn] variants: $VARIANTS"
echo "[run-post-quant-ffn] active_experts: $ACTIVE_EXPERTS"
echo "[run-post-quant-ffn] tokens_per_expert: $TOKENS_PER_EXPERT"
echo "[run-post-quant-ffn] total_tokens: $TOTAL_TOKENS"
echo "[run-post-quant-ffn] main_weight_group_size=$MAIN_WEIGHT_GROUP_SIZE ose_weight_group_size=$OSE_WEIGHT_GROUP_SIZE ose_topk=$OSE_TOPK"
echo "[run-post-quant-ffn] warmup=$WARMUP repeat=$REPEAT rounds=$ROUNDS validate=$VALIDATE"

python3 tools/triton_post_quant_moe_ffn_duquant_vs_ose_benchmark.py \
  --output_dir "$OUT_ROOT" \
  --scope "$SCOPE" \
  --presets "$PRESETS" \
  --variants "$VARIANTS" \
  --active_experts "$ACTIVE_EXPERTS" \
  --tokens_per_expert "$TOKENS_PER_EXPERT" \
  --total_tokens "$TOTAL_TOKENS" \
  --ose_topk "$OSE_TOPK" \
  --main_weight_group_size "$MAIN_WEIGHT_GROUP_SIZE" \
  --ose_weight_group_size "$OSE_WEIGHT_GROUP_SIZE" \
  --duquant_block_size "$DUQUANT_BLOCK_SIZE" \
  --duquant_rotation_stages "$DUQUANT_ROTATION_STAGES" \
  "${perm_arg[@]}" \
  "${down_arg[@]}" \
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

python3 tools/plot_post_quant_moe_ffn_duquant_vs_ose_benchmark.py \
  --input_csv "$OUT_ROOT/post_quant_moe_ffn_duquant_vs_ose_detail.csv" \
  --output_dir "$OUT_ROOT/plots"

echo "[run-post-quant-ffn] done"
echo "[run-post-quant-ffn] detail: $OUT_ROOT/post_quant_moe_ffn_duquant_vs_ose_detail.csv"
echo "[run-post-quant-ffn] report: $OUT_ROOT/report.md"
echo "[run-post-quant-ffn] plots: $OUT_ROOT/plots"
