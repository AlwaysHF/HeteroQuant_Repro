#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

MODEL="${MODEL:-$ROOT/local_models/olmoe_compat}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
PLAN_PATH="${PLAN_PATH-$ROOT/plans/olmoe_routed_range/moe_quant_plan_routed_range.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/repro_olmoe_691_$(date +%Y%m%d_%H%M%S)/heteroquant_a8w4}"
TEST_DATASET="${TEST_DATASET-wikitext2,c4}"
TASKS="${TASKS-piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
FAST_MOE_DOWN_CALIBRATION="${FAST_MOE_DOWN_CALIBRATION:-1}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-8}"
ROUTER_W_BITS="${ROUTER_W_BITS:-8}"
ROUTER_A_BITS="${ROUTER_A_BITS:-8}"
MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-w8a8}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"
A_DYNAMIC_METHOD="${A_DYNAMIC_METHOD:-per_token}"
FC1_SCALE_MERGE="${FC1_SCALE_MERGE:-act_p99}"
ACT_MEAN_BETA="${ACT_MEAN_BETA:-1}"
MOE_DOWN_SMOOTH_MODE="${MOE_DOWN_SMOOTH_MODE:-otsu}"
LAC="${LAC:-}"
SWC="${SWC:-}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-1}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"
DISABLE_MOE_GATE_UP_DUQUANT_ROTATION="${DISABLE_MOE_GATE_UP_DUQUANT_ROTATION:-1}"

cd "$ROOT"

task_args=()
if [[ -n "$TASKS" ]]; then
  task_args=(--tasks "$TASKS" --num_fewshot "$NUM_FEWSHOT" --batch_size "$BATCH_SIZE")
fi

gate_up_rotation_args=()
if [[ "$DISABLE_MOE_GATE_UP_DUQUANT_ROTATION" == "1" || "$DISABLE_MOE_GATE_UP_DUQUANT_ROTATION" == "true" || "$DISABLE_MOE_GATE_UP_DUQUANT_ROTATION" == "TRUE" ]]; then
  gate_up_rotation_args=(--disable_moe_gate_up_duquant_rotation)
fi

fast_moe_args=()
if [[ "$FAST_MOE_DOWN_CALIBRATION" == "1" || "$FAST_MOE_DOWN_CALIBRATION" == "true" || "$FAST_MOE_DOWN_CALIBRATION" == "TRUE" ]]; then
  fast_moe_args=(--fast_moe_down_calibration --fast_moe_calib_tokens "$FAST_MOE_CALIB_TOKENS")
fi

clip_args=()
if [[ -n "$LAC" ]]; then
  clip_args+=(--lac "$LAC")
fi
if [[ -n "$SWC" ]]; then
  clip_args+=(--swc "$SWC")
fi

python3 main.py \
  --model "$MODEL" \
  --model_name olmoe \
  --wbits "$W_BITS" --abits "$A_BITS" \
  --router_wbits "$ROUTER_W_BITS" --router_abits "$ROUTER_A_BITS" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --w_dynamic_method per_channel_tensor \
  --a_dynamic_method "$A_DYNAMIC_METHOD" \
  --router_w_dynamic_method per_channel_kl_top0 \
  --nsamples "$NSAMPLES" \
  --seq_length "$SEQ_LENGTH" \
  --scale_search_steps "$SCALE_SEARCH_STEPS" \
  --eval_ppl \
  --test_dataset "$TEST_DATASET" \
  "${task_args[@]}" \
  --group_size -1 \
  --weight_channel_group_size "$WEIGHT_CHANNEL_GROUP_SIZE" \
  --router_weight_channel_group_size "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
  --act_group_size -1 \
  --smooth \
  --fc1_scale_merge "$FC1_SCALE_MERGE" \
  --act_mean_beta "$ACT_MEAN_BETA" \
  --moe_down_smooth_mode "$MOE_DOWN_SMOOTH_MODE" \
  --moe_outlier_topk "$MOE_OUTLIER_TOPK" \
  --moe_outlier_quant "$MOE_OUTLIER_QUANT" \
  --moe_outlier_score "$MOE_OUTLIER_SCORE" \
  "${clip_args[@]}" \
  "${gate_up_rotation_args[@]}" \
  --moe_quant_plan "$PLAN_PATH" \
  "${fast_moe_args[@]}"
