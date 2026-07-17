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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/ose_precision_routed_tail_olmoe_ppl_group2048_${RUN_ID}}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-2048}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-2100}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"
TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
TIER_FRACTIONS="${TIER_FRACTIONS:-0.234375,0.59375,0.171875}"
QUANTS="${QUANTS:-fp16,w16a16,w8a8,w4a4,same}"

mkdir -p "$CACHE_DIR"
planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[ose-precision-routed-tail-2048] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

mkdir -p "$OUT_ROOT"
PLAN_DIR="$OUT_ROOT/plans"
mkdir -p "$PLAN_DIR"
PLAN_LOG="$OUT_ROOT/plan.log"
PLAN_PATH="$PLAN_DIR/moe_quant_plan_routed_tail.json"
SUMMARY_PATH="$PLAN_DIR/moe_quant_plan_routed_tail_summary.json"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'label,moe_outlier_quant,moe_outlier_topk,moe_outlier_score,metric,tier_quants,tier_fractions,weight_channel_group_size,router_weight_channel_group_size,actual_avg_sum_bits,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,plan_path,output_dir,run_log\n' > "$SUMMARY_CSV"

IFS=',' read -r -a QUANT_LIST <<< "$QUANTS"

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
  local label="$1" quant="$2" actual_avg_sum_bits="$3" status="$4" wikitext2="$5" c4="$6"
  local calibration_seconds="$7" duquant_seconds="$8" inference_wikitext2_seconds="$9"
  local inference_c4_seconds="${10}" output_dir="${11}" run_log="${12}"
  {
    csv_quote "$label"; printf ','
    csv_quote "$quant"; printf ','
    csv_quote "$MOE_OUTLIER_TOPK"; printf ','
    csv_quote "$MOE_OUTLIER_SCORE"; printf ','
    csv_quote "routed_tail"; printf ','
    csv_quote "$TIER_QUANTS"; printf ','
    csv_quote "$TIER_FRACTIONS"; printf ','
    csv_quote "$WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$actual_avg_sum_bits"; printf ','
    csv_quote "$status"; printf ','
    csv_quote "$wikitext2"; printf ','
    csv_quote "$c4"; printf ','
    csv_quote "$calibration_seconds"; printf ','
    csv_quote "$duquant_seconds"; printf ','
    csv_quote "$inference_wikitext2_seconds"; printf ','
    csv_quote "$inference_c4_seconds"; printf ','
    csv_quote "$PLAN_PATH"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$run_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

echo "[ose-precision-routed-tail-2048] output root: $OUT_ROOT"
echo "[ose-precision-routed-tail-2048] quants: $QUANTS"
echo "[ose-precision-routed-tail-2048] metric: routed_tail"
echo "[ose-precision-routed-tail-2048] topk: $MOE_OUTLIER_TOPK"
echo "[ose-precision-routed-tail-2048] tier: $TIER_QUANTS / $TIER_FRACTIONS"

planner_status="ok"
if ! run_logged "$PLAN_LOG" \
  python3 tools/olmoe_expert_distribution_planner.py \
    --model "$MODEL" \
    --model_name olmoe \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$PLAN_DIR" \
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
    --moe_outlier_topk "$MOE_OUTLIER_TOPK" \
    --moe_outlier_score "$MOE_OUTLIER_SCORE" \
    --smooth \
    --fc1_scale_merge act_p99 \
    --act_mean_beta 1 \
    --reuse_dataloader_cache; then
  planner_status="plan_failed"
fi

total="${#QUANT_LIST[@]}"
start_ts="$(date +%s)"

for idx in "${!QUANT_LIST[@]}"; do
  quant="$(echo "${QUANT_LIST[$idx]}" | xargs)"
  [[ -n "$quant" ]] || continue
  case_num=$((idx + 1))
  label="ose_${quant}"
  output_dir="$OUT_ROOT/$label/output"
  run_log="$OUT_ROOT/$label/run.log"
  mkdir -p "$output_dir"

  echo
  echo "[ose-precision-routed-tail-2048] case $case_num/$total: $label"

  status="$planner_status"
  if [[ "$status" == "ok" && ! -f "$PLAN_PATH" ]]; then
    status="missing_plan"
  fi
  if [[ "$status" == "ok" ]]; then
    if ! run_logged "$run_log" \
      env PLAN_PATH="$PLAN_PATH" \
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
          MOE_OUTLIER_QUANT="$quant" \
          MOE_OUTLIER_SCORE="$MOE_OUTLIER_SCORE" \
          WEIGHT_CHANNEL_GROUP_SIZE="$WEIGHT_CHANNEL_GROUP_SIZE" \
          ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
          bash scripts/reproduce_olmoe_691.sh; then
      status="run_failed"
    fi
  fi

  rank_log="$(find "$output_dir" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
  actual_avg_sum_bits=""
  if [[ -f "$SUMMARY_PATH" ]]; then
    actual_avg_sum_bits="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("actual_avg_sum_bits",""))' "$SUMMARY_PATH")"
  fi
  wikitext2="$(extract_last_value 'wikitext2 :' "$rank_log")"
  c4="$(extract_last_value 'c4 :' "$rank_log")"
  calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
  duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
  inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
  inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"

  append_summary "$label" "$quant" "$actual_avg_sum_bits" "$status" "$wikitext2" "$c4" \
    "$calibration_seconds" "$duquant_seconds" "$inference_wikitext2_seconds" "$inference_c4_seconds" \
    "$output_dir" "$run_log"

  now_ts="$(date +%s)"
  remaining_cases=$((total - case_num))
  eta="$(date -d "@$((now_ts + remaining_cases * EST_SECONDS_PER_RUN))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  echo "[ose-precision-routed-tail-2048] finished $label with status=$status eta=${eta:-unknown}"
  echo "[ose-precision-routed-tail-2048] summary: $SUMMARY_CSV"
done

echo
echo "[ose-precision-routed-tail-2048] all done: $SUMMARY_CSV"
