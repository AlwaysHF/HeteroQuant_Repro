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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/tier_set_retest_olmoe_$(date +%Y%m%d_%H%M%S)}"
TEST_DATASET="${TEST_DATASET:-wikitext2}"
TASKS="${TASKS:-}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-900}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

mkdir -p "$CACHE_DIR"
planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[tier-set] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'label,tier_quants,tier_fractions,actual_avg_sum_bits,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,plan_path,output_dir,case_log\n' > "$SUMMARY_CSV"

CASES=(
  "a8w8|w4g-1-a4g-1,w8g-1-a8g-1|0.5,0.5"
  "a8w6|w4g-1-a4g-1,w6g-1-a8g-1|0.333333333333,0.666666666667"
  "a8w5|w4g-1-a4g-1,w5g-1-a8g-1|0.2,0.8"
  "a8w5_a8w6|w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1|0.234375,0.59375,0.171875"
  "a8w4_a8w5_a8w6|w4g-1-a4g-1,w4g-1-a8g-1,w5g-1-a8g-1,w6g-1-a8g-1|0.2,0.1,0.6,0.1"
  "a8w4_a8w5_a8w6_a8w8|w4g-1-a4g-1,w4g-1-a8g-1,w5g-1-a8g-1,w6g-1-a8g-1,w8g-1-a8g-1|0.15,0.5,0.2,0.1,0.05"
)

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
  local tier_quants="$2"
  local tier_fractions="$3"
  local actual_avg_sum_bits="$4"
  local status="$5"
  local wikitext2="$6"
  local c4="$7"
  local calibration_seconds="$8"
  local duquant_seconds="$9"
  local inference_wikitext2_seconds="${10}"
  local plan_path="${11}"
  local output_dir="${12}"
  local case_log="${13}"

  {
    csv_quote "$label"; printf ','
    csv_quote "$tier_quants"; printf ','
    csv_quote "$tier_fractions"; printf ','
    csv_quote "$actual_avg_sum_bits"; printf ','
    csv_quote "$status"; printf ','
    csv_quote "$wikitext2"; printf ','
    csv_quote "$c4"; printf ','
    csv_quote "$calibration_seconds"; printf ','
    csv_quote "$duquant_seconds"; printf ','
    csv_quote "$inference_wikitext2_seconds"; printf ','
    csv_quote "$plan_path"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$case_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

total="${#CASES[@]}"
start_ts="$(date +%s)"

echo "[tier-set] output root: $OUT_ROOT"
echo "[tier-set] test_dataset: $TEST_DATASET"
echo "[tier-set] tasks: ${TASKS:-<none>}"
echo "[tier-set] scale_search_steps: $SCALE_SEARCH_STEPS"

for idx in "${!CASES[@]}"; do
  IFS='|' read -r label tier_quants tier_fractions <<< "${CASES[$idx]}"
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
  echo "[tier-set] case $case_num/$total: $label"
  echo "[tier-set] tier_quants: $tier_quants"
  echo "[tier-set] tier_fractions: $tier_fractions"

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
      --tier_quants "$tier_quants" \
      --tier_fractions "$tier_fractions" \
      --route_alpha 0.5 \
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

  append_summary "$label" "$tier_quants" "$tier_fractions" "$actual_avg_sum_bits" "$status" \
    "$wikitext2" "$c4" "$calibration_seconds" "$duquant_seconds" "$inference_wikitext2_seconds" \
    "$plan_path" "$output_dir" "$run_log"

  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  remaining_cases=$((total - case_num))
  est_remaining=$((remaining_cases * EST_SECONDS_PER_RUN))
  eta="$(date -d "@$((now_ts + est_remaining))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  echo "[tier-set] finished $label with status=$status"
  echo "[tier-set] elapsed=${elapsed}s, estimated_remaining=${est_remaining}s, eta=${eta:-unknown}"
  echo "[tier-set] summary: $SUMMARY_CSV"
done

echo
echo "[tier-set] all done: $SUMMARY_CSV"
