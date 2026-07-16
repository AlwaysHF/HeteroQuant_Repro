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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/allocation_component_ablation_olmoe_ppl_$(date +%Y%m%d_%H%M%S)}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-1}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"
INCLUDE_SINGLE_TERM="${INCLUDE_SINGLE_TERM:-0}"
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-900}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
TIER_FRACTIONS="${TIER_FRACTIONS:-0.234375,0.59375,0.171875}"

mkdir -p "$CACHE_DIR"
planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[component-ablation] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'label,input_score_coef,down_score_coef,weight_score_coef,score_transform,route_alpha,tier_quants,tier_fractions,actual_avg_sum_bits,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,plan_path,output_dir,case_log\n' > "$SUMMARY_CSV"

CASES=(
  "route_range|1.0|1.0|1.0"
  "no_input_range|0.0|1.0|1.0"
  "no_down_range|1.0|0.0|1.0"
  "no_weight_range|1.0|1.0|0.0"
)

if [[ "$INCLUDE_SINGLE_TERM" == "1" || "$INCLUDE_SINGLE_TERM" == "true" || "$INCLUDE_SINGLE_TERM" == "TRUE" ]]; then
  CASES+=(
    "input_only|1.0|0.0|0.0"
    "down_only|0.0|1.0|0.0"
    "weight_only|0.0|0.0|1.0"
  )
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

append_summary() {
  local label="$1"
  local input_score_coef="$2"
  local down_score_coef="$3"
  local weight_score_coef="$4"
  local actual_avg_sum_bits="$5"
  local status="$6"
  local wikitext2="$7"
  local c4="$8"
  local calibration_seconds="$9"
  local duquant_seconds="${10}"
  local inference_wikitext2_seconds="${11}"
  local inference_c4_seconds="${12}"
  local plan_path="${13}"
  local output_dir="${14}"
  local case_log="${15}"

  {
    csv_quote "$label"; printf ','
    csv_quote "$input_score_coef"; printf ','
    csv_quote "$down_score_coef"; printf ','
    csv_quote "$weight_score_coef"; printf ','
    csv_quote "raw"; printf ','
    csv_quote "1.0"; printf ','
    csv_quote "$TIER_QUANTS"; printf ','
    csv_quote "$TIER_FRACTIONS"; printf ','
    csv_quote "$actual_avg_sum_bits"; printf ','
    csv_quote "$status"; printf ','
    csv_quote "$wikitext2"; printf ','
    csv_quote "$c4"; printf ','
    csv_quote "$calibration_seconds"; printf ','
    csv_quote "$duquant_seconds"; printf ','
    csv_quote "$inference_wikitext2_seconds"; printf ','
    csv_quote "$inference_c4_seconds"; printf ','
    csv_quote "$plan_path"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$case_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

total="${#CASES[@]}"
start_ts="$(date +%s)"

echo "[component-ablation] output root: $OUT_ROOT"
echo "[component-ablation] test_dataset: $TEST_DATASET"
echo "[component-ablation] tasks: ${TASKS:-<none>}"
echo "[component-ablation] tier_quants: $TIER_QUANTS"
echo "[component-ablation] tier_fractions: $TIER_FRACTIONS"
echo "[component-ablation] formula: (input_coef*input_range + down_coef*down_range + weight_coef*weight_range) * route_norm"

for idx in "${!CASES[@]}"; do
  IFS='|' read -r label input_score_coef down_score_coef weight_score_coef <<< "${CASES[$idx]}"
  case_num=$((idx + 1))
  case_dir="$OUT_ROOT/$label"
  plan_dir="$case_dir/plans"
  output_dir="$case_dir/output"
  plan_log="$case_dir/plan.log"
  run_log="$case_dir/run.log"
  plan_path="$plan_dir/moe_quant_plan_routed_range.json"
  summary_path="$plan_dir/moe_quant_plan_routed_range_summary.json"

  mkdir -p "$plan_dir" "$output_dir"

  echo
  echo "[component-ablation] case $case_num/$total: $label"
  echo "[component-ablation] input=$input_score_coef down=$down_score_coef weight=$weight_score_coef"

  status="ok"
  if ! run_logged "$plan_log" \
    python3 tools/olmoe_expert_distribution_planner.py \
      --model "$MODEL" \
      --model_name olmoe \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$plan_dir" \
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
      --input_score_coef "$input_score_coef" \
      --down_score_coef "$down_score_coef" \
      --weight_score_coef "$weight_score_coef" \
      --moe_outlier_topk 64 \
      --moe_outlier_score smooth_scale \
      --smooth \
      --fc1_scale_merge act_p99 \
      --act_mean_beta 1 \
      --reuse_dataloader_cache; then
    status="plan_failed"
  fi

  if [[ "$status" == "ok" ]]; then
    if ! run_logged "$run_log" \
      env PLAN_PATH="$plan_path" \
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
          bash scripts/reproduce_olmoe_691.sh; then
      status="run_failed"
    fi
  fi

  rank_log="$(find "$output_dir" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
  actual_avg_sum_bits=""
  if [[ -f "$summary_path" ]]; then
    actual_avg_sum_bits="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("actual_avg_sum_bits",""))' "$summary_path")"
  fi
  wikitext2="$(extract_last_value 'wikitext2 :' "$rank_log")"
  c4="$(extract_last_value 'c4 :' "$rank_log")"
  calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
  duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
  inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
  inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"

  append_summary "$label" "$input_score_coef" "$down_score_coef" "$weight_score_coef" "$actual_avg_sum_bits" "$status" \
    "$wikitext2" "$c4" "$calibration_seconds" "$duquant_seconds" \
    "$inference_wikitext2_seconds" "$inference_c4_seconds" "$plan_path" "$output_dir" "$run_log"

  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  remaining_cases=$((total - case_num))
  est_remaining=$((remaining_cases * EST_SECONDS_PER_RUN))
  eta="$(date -d "@$((now_ts + est_remaining))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  echo "[component-ablation] finished $label with status=$status"
  echo "[component-ablation] elapsed=${elapsed}s, estimated_remaining=${est_remaining}s, eta=${eta:-unknown}"
  echo "[component-ablation] summary: $SUMMARY_CSV"
done

echo
echo "[component-ablation] all done: $SUMMARY_CSV"
