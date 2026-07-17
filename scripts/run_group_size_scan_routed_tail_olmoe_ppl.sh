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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/group_size_scan_routed_tail_olmoe_ppl_${RUN_ID}}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-1500}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

GROUP_SIZES="${GROUP_SIZES:-1,16,32,64,128,256,512,1024,2048}"
MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-w8a8}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"
TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
TIER_FRACTIONS="${TIER_FRACTIONS:-0.234375,0.59375,0.171875}"

mkdir -p "$CACHE_DIR"
planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[group-scan-routed-tail] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'group_size,metric,tier_quants,tier_fractions,moe_outlier_topk,moe_outlier_quant,moe_outlier_score,actual_avg_sum_bits,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,plan_path,output_dir,plan_log,run_log\n' > "$SUMMARY_CSV"

IFS=',' read -r -a GROUP_LIST <<< "$GROUP_SIZES"

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
  local group_size="$1"
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
  local plan_log="${12}"
  local run_log="${13}"

  {
    csv_quote "$group_size"; printf ','
    csv_quote "routed_tail"; printf ','
    csv_quote "$TIER_QUANTS"; printf ','
    csv_quote "$TIER_FRACTIONS"; printf ','
    csv_quote "$MOE_OUTLIER_TOPK"; printf ','
    csv_quote "$MOE_OUTLIER_QUANT"; printf ','
    csv_quote "$MOE_OUTLIER_SCORE"; printf ','
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
    csv_quote "$plan_log"; printf ','
    csv_quote "$run_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

total="${#GROUP_LIST[@]}"
start_ts="$(date +%s)"

echo "[group-scan-routed-tail] output root: $OUT_ROOT"
echo "[group-scan-routed-tail] group_sizes: $GROUP_SIZES"
echo "[group-scan-routed-tail] test_dataset: $TEST_DATASET"
echo "[group-scan-routed-tail] tasks: ${TASKS:-<none>}"
echo "[group-scan-routed-tail] metric: routed_tail"
echo "[group-scan-routed-tail] tier: $TIER_QUANTS / $TIER_FRACTIONS"
echo "[group-scan-routed-tail] OSE: topk=$MOE_OUTLIER_TOPK quant=$MOE_OUTLIER_QUANT score=$MOE_OUTLIER_SCORE"

for idx in "${!GROUP_LIST[@]}"; do
  group_size="$(echo "${GROUP_LIST[$idx]}" | xargs)"
  [[ -n "$group_size" ]] || continue
  case_num=$((idx + 1))
  label="group_${group_size}"
  case_dir="$OUT_ROOT/$label"
  plan_dir="$case_dir/plans"
  output_dir="$case_dir/output"
  plan_log="$case_dir/plan.log"
  run_log="$case_dir/run.log"
  plan_path="$plan_dir/moe_quant_plan_routed_tail.json"
  summary_path="$plan_dir/moe_quant_plan_routed_tail_summary.json"

  mkdir -p "$plan_dir" "$output_dir"

  echo
  echo "[group-scan-routed-tail] case $case_num/$total: group_size=$group_size"

  status="ok"
  if ! run_logged "$plan_log" \
    python3 tools/olmoe_expert_distribution_planner.py \
      --model "$MODEL" \
      --model_name olmoe \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$plan_dir" \
      --metrics routed_tail \
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
      --weight_channel_group_size "$group_size" \
      --moe_outlier_topk "$MOE_OUTLIER_TOPK" \
      --moe_outlier_score "$MOE_OUTLIER_SCORE" \
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
          MOE_OUTLIER_TOPK="$MOE_OUTLIER_TOPK" \
          MOE_OUTLIER_QUANT="$MOE_OUTLIER_QUANT" \
          MOE_OUTLIER_SCORE="$MOE_OUTLIER_SCORE" \
          WEIGHT_CHANNEL_GROUP_SIZE="$group_size" \
          ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="$group_size" \
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

  append_summary "$group_size" "$actual_avg_sum_bits" "$status" "$wikitext2" "$c4" \
    "$calibration_seconds" "$duquant_seconds" "$inference_wikitext2_seconds" "$inference_c4_seconds" \
    "$plan_path" "$output_dir" "$plan_log" "$run_log"

  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  remaining_cases=$((total - case_num))
  est_remaining=$((remaining_cases * EST_SECONDS_PER_RUN))
  eta="$(date -d "@$((now_ts + est_remaining))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  echo "[group-scan-routed-tail] finished group_size=$group_size with status=$status"
  echo "[group-scan-routed-tail] elapsed=${elapsed}s, estimated_remaining=${est_remaining}s, eta=${eta:-unknown}"
  echo "[group-scan-routed-tail] summary: $SUMMARY_CSV"
done

echo
echo "[group-scan-routed-tail] all done: $SUMMARY_CSV"
