#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

MODEL="${MODEL:-$ROOT/local_models/olmoe_compat}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/ose_selection_metric_ablation_olmoe_ppl_${RUN_ID}}"

TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"

WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-1}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"
MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-w8a8}"
SELECTION_SCORES="${SELECTION_SCORES:-smooth_scale,shared_scale,expert_max,expert_error,expert_p99_mean}"

TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
TIER_FRACTIONS="${TIER_FRACTIONS:-0.234375,0.59375,0.171875}"

DEFAULT_COMMON_PLAN="$ROOT/experiments/route_range_topk_group_ablation_olmoe_ppl_20260721_233425/group_1/plans/moe_quant_plan_routed_range.json"
COMMON_PLAN="${COMMON_PLAN:-$DEFAULT_COMMON_PLAN}"
PLAN_SCORE="${PLAN_SCORE:-shared_scale}"
PLAN_DIR="$OUT_ROOT/common_plan"
PLAN_LOG="$PLAN_DIR/plan.log"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

mkdir -p "$CACHE_DIR" "$OUT_ROOT"

planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[ose-selection] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

run_logged() {
  local logfile="$1"
  shift
  set +e
  "$@" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

extract_last_value() {
  local pattern="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    grep -E "$pattern" "$file" | tail -n 1 | sed -E 's/.*: *([0-9.]+).*/\1/' || true
  fi
}

csv_quote() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

sanitize_label() {
  printf '%s' "$1" | tr '/ ' '__'
}

if [[ ! -f "$COMMON_PLAN" ]]; then
  COMMON_PLAN="$PLAN_DIR/moe_quant_plan_routed_range.json"
  mkdir -p "$PLAN_DIR"
  echo "[ose-selection] common plan not found; generate shared route_range plan at $COMMON_PLAN"
  run_logged "$PLAN_LOG" \
    python3 tools/olmoe_expert_distribution_planner.py \
      --model "$MODEL" \
      --model_name olmoe \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$PLAN_DIR" \
      --metrics routed_range \
      --calib_dataset wikitext2 \
      --nsamples "$NSAMPLES_PLAN" \
      --seq_length "$SEQ_LENGTH" \
      --seed 2 \
      --attn_implementation eager \
      --selection_scope per_layer \
      --plan_granularity expert \
      --target_avg_sum_bits 12.0 \
      --tier_quants "$TIER_QUANTS" \
      --tier_fractions "$TIER_FRACTIONS" \
      --route_alpha 1.0 \
      --score_transform raw \
      --input_score_coef 1.0 \
      --down_score_coef 1.0 \
      --weight_score_coef 1.0 \
      --weight_score_projs gate_proj,up_proj,down_proj \
      --weight_channel_group_size "$WEIGHT_CHANNEL_GROUP_SIZE" \
      --moe_outlier_topk "$MOE_OUTLIER_TOPK" \
      --moe_outlier_score "$PLAN_SCORE" \
      --smooth \
      --fc1_scale_merge act_p99 \
      --act_mean_beta 1 \
      --reuse_dataloader_cache
else
  echo "[ose-selection] reuse common plan: $COMMON_PLAN"
fi

SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'selection_label,moe_outlier_score,planner_metric,common_plan,topk,moe_outlier_quant,weight_channel_group_size,router_weight_channel_group_size,tier_quants,tier_fractions,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,output_dir,run_log,rank_log\n' > "$SUMMARY_CSV"

append_summary() {
  local selection_label="$1"
  local score="$2"
  local status="$3"
  local wikitext2="$4"
  local c4="$5"
  local calibration_seconds="$6"
  local duquant_seconds="$7"
  local inference_wikitext2_seconds="$8"
  local inference_c4_seconds="$9"
  local output_dir="${10}"
  local run_log="${11}"
  local rank_log="${12}"

  {
    csv_quote "$selection_label"; printf ','
    csv_quote "$score"; printf ','
    csv_quote "route_range"; printf ','
    csv_quote "$COMMON_PLAN"; printf ','
    csv_quote "$MOE_OUTLIER_TOPK"; printf ','
    csv_quote "$MOE_OUTLIER_QUANT"; printf ','
    csv_quote "$WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$TIER_QUANTS"; printf ','
    csv_quote "$TIER_FRACTIONS"; printf ','
    csv_quote "$status"; printf ','
    csv_quote "$wikitext2"; printf ','
    csv_quote "$c4"; printf ','
    csv_quote "$calibration_seconds"; printf ','
    csv_quote "$duquant_seconds"; printf ','
    csv_quote "$inference_wikitext2_seconds"; printf ','
    csv_quote "$inference_c4_seconds"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$run_log"; printf ','
    csv_quote "$rank_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

IFS=',' read -r -a SCORE_LIST <<< "$SELECTION_SCORES"
echo "[ose-selection] output root: $OUT_ROOT"
echo "[ose-selection] scores: $SELECTION_SCORES"
echo "[ose-selection] fixed plan: $COMMON_PLAN"
echo "[ose-selection] fixed config: route_range, group=$WEIGHT_CHANNEL_GROUP_SIZE, topK=$MOE_OUTLIER_TOPK, OSE=$MOE_OUTLIER_QUANT"
echo "[ose-selection] datasets: $TEST_DATASET"

for score_raw in "${SCORE_LIST[@]}"; do
  score="$(echo "$score_raw" | xargs)"
  [[ -n "$score" ]] || continue
  label="$(sanitize_label "$score")"
  case_dir="$OUT_ROOT/$label"
  output_dir="$case_dir/output"
  run_log="$case_dir/run.log"
  mkdir -p "$output_dir"

  echo
  echo "[ose-selection] run selection=$score"
  status="ok"
  if ! run_logged "$run_log" \
    env PLAN_PATH="$COMMON_PLAN" \
        OUTPUT_DIR="$output_dir" \
        MODEL="$MODEL" \
        CACHE_DIR="$CACHE_DIR" \
        TEST_DATASET="$TEST_DATASET" \
        TASKS="$TASKS" \
        NSAMPLES="$NSAMPLES_RUN" \
        SEQ_LENGTH="$SEQ_LENGTH" \
        SCALE_SEARCH_STEPS="$SCALE_SEARCH_STEPS" \
        FAST_MOE_CALIB_TOKENS="$FAST_MOE_CALIB_TOKENS" \
        WEIGHT_CHANNEL_GROUP_SIZE="$WEIGHT_CHANNEL_GROUP_SIZE" \
        ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
        MOE_OUTLIER_TOPK="$MOE_OUTLIER_TOPK" \
        MOE_OUTLIER_QUANT="$MOE_OUTLIER_QUANT" \
        MOE_OUTLIER_SCORE="$score" \
        DISABLE_MOE_GATE_UP_DUQUANT_ROTATION=1 \
        bash scripts/reproduce_olmoe_691.sh; then
    status="run_failed"
  fi

  rank_log="$(find "$output_dir" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
  wikitext2="$(extract_last_value 'wikitext2 :' "$rank_log")"
  c4="$(extract_last_value 'c4 :' "$rank_log")"
  calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
  duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
  inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
  inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"

  append_summary "$label" "$score" "$status" "$wikitext2" "$c4" \
    "$calibration_seconds" "$duquant_seconds" \
    "$inference_wikitext2_seconds" "$inference_c4_seconds" \
    "$output_dir" "$run_log" "$rank_log"
  echo "[ose-selection] finished selection=$score status=$status summary=$SUMMARY_CSV"
done

echo
echo "[ose-selection] all done: $SUMMARY_CSV"
