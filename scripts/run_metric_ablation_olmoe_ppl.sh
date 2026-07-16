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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/metric_ablation_olmoe_ppl_$(date +%Y%m%d_%H%M%S)}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-1}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-900}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

METRICS="${METRICS:-residual_qerr_gain,residual_qerr,kurtosis,route,tail}"
TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
TIER_FRACTIONS="${TIER_FRACTIONS:-0.234375,0.59375,0.171875}"
WEIGHT_SCORE_PROJS="${WEIGHT_SCORE_PROJS:-gate_proj,up_proj,down_proj}"

mkdir -p "$CACHE_DIR"
planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[metric-ablation] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

mkdir -p "$OUT_ROOT"
PLAN_DIR="$OUT_ROOT/plans"
mkdir -p "$PLAN_DIR"
PLANNER_LOG="$OUT_ROOT/planner.log"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'metric,actual_avg_sum_bits,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,plan_path,output_dir,run_log\n' > "$SUMMARY_CSV"

IFS=',' read -r -a METRIC_LIST <<< "$METRICS"

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
  local metric="$1"
  local actual_avg_sum_bits="$2"
  local status="$3"
  local wikitext2="$4"
  local c4="$5"
  local calibration_seconds="$6"
  local duquant_seconds="$7"
  local inference_wikitext2_seconds="$8"
  local inference_c4_seconds="$9"
  local plan_path="${10}"
  local output_dir="${11}"
  local run_log="${12}"

  {
    csv_quote "$metric"; printf ','
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
    csv_quote "$run_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

echo "[metric-ablation] output root: $OUT_ROOT"
echo "[metric-ablation] metrics: $METRICS"
echo "[metric-ablation] test_dataset: $TEST_DATASET"
echo "[metric-ablation] tasks: ${TASKS:-<none>}"
echo "[metric-ablation] tier_quants: $TIER_QUANTS"
echo "[metric-ablation] tier_fractions: $TIER_FRACTIONS"

planner_status="ok"
if ! run_logged "$PLANNER_LOG" \
  python3 tools/olmoe_expert_distribution_planner.py \
    --model "$MODEL" \
    --model_name olmoe \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$PLAN_DIR" \
    --metrics "$METRICS" \
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
    --weight_score_projs "$WEIGHT_SCORE_PROJS" \
    --moe_outlier_topk 64 \
    --moe_outlier_score smooth_scale \
    --smooth \
    --fc1_scale_merge act_p99 \
    --act_mean_beta 1 \
    --reuse_dataloader_cache; then
  planner_status="plan_failed"
fi

total="${#METRIC_LIST[@]}"
start_ts="$(date +%s)"

for idx in "${!METRIC_LIST[@]}"; do
  metric="$(echo "${METRIC_LIST[$idx]}" | xargs)"
  [[ -n "$metric" ]] || continue
  case_num=$((idx + 1))
  output_dir="$OUT_ROOT/$metric/output"
  run_log="$OUT_ROOT/$metric/run.log"
  plan_path="$PLAN_DIR/moe_quant_plan_${metric}.json"
  summary_path="$PLAN_DIR/moe_quant_plan_${metric}_summary.json"
  mkdir -p "$output_dir"

  echo
  echo "[metric-ablation] case $case_num/$total: $metric"

  status="$planner_status"
  if [[ "$status" == "ok" && ! -f "$plan_path" ]]; then
    status="missing_plan"
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

  append_summary "$metric" "$actual_avg_sum_bits" "$status" "$wikitext2" "$c4" \
    "$calibration_seconds" "$duquant_seconds" "$inference_wikitext2_seconds" "$inference_c4_seconds" \
    "$plan_path" "$output_dir" "$run_log"

  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  remaining_cases=$((total - case_num))
  est_remaining=$((remaining_cases * EST_SECONDS_PER_RUN))
  eta="$(date -d "@$((now_ts + est_remaining))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  echo "[metric-ablation] finished $metric with status=$status"
  echo "[metric-ablation] elapsed=${elapsed}s, estimated_remaining=${est_remaining}s, eta=${eta:-unknown}"
  echo "[metric-ablation] summary: $SUMMARY_CSV"
done

echo
echo "[metric-ablation] all done: $SUMMARY_CSV"
