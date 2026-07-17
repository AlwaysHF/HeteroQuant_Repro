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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/component_ablation_routed_tail_olmoe_ppl_group2048_${RUN_ID}}"
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

UNIFORM_W4A8_QUANTS="${UNIFORM_W4A8_QUANTS:-w4g-1-a8g-1,w4g-1-a8g-1}"
UNIFORM_W4A8_FRACTIONS="${UNIFORM_W4A8_FRACTIONS:-1.0,0.0}"
FINAL_TIER_QUANTS="${FINAL_TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
FINAL_TIER_FRACTIONS="${FINAL_TIER_FRACTIONS:-0.234375,0.59375,0.171875}"

mkdir -p "$CACHE_DIR"
planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[component-routed-tail-2048] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'label,ose,allocation,planner_metric,moe_outlier_topk,moe_outlier_quant,moe_outlier_score,tier_quants,tier_fractions,disable_gateup_duquant,weight_channel_group_size,router_weight_channel_group_size,actual_avg_sum_bits,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,plan_path,output_dir,plan_log,run_log\n' > "$SUMMARY_CSV"

CASES=(
  "no_gateup_rotation|0|0|routed_tail|0|same|smooth_scale|$UNIFORM_W4A8_QUANTS|$UNIFORM_W4A8_FRACTIONS|1"
  "ose_only|1|0|routed_tail|64|w8a8|smooth_scale|$UNIFORM_W4A8_QUANTS|$UNIFORM_W4A8_FRACTIONS|1"
  "allocation_only|0|1|routed_tail|0|same|smooth_scale|$FINAL_TIER_QUANTS|$FINAL_TIER_FRACTIONS|1"
  "ose_random_same_budget|1|1|random|64|w8a8|smooth_scale|$FINAL_TIER_QUANTS|$FINAL_TIER_FRACTIONS|1"
  "full_method|1|1|routed_tail|64|w8a8|smooth_scale|$FINAL_TIER_QUANTS|$FINAL_TIER_FRACTIONS|1"
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
  local label="$1" ose="$2" allocation="$3" planner_metric="$4" topk="$5" outlier_quant="$6" outlier_score="$7"
  local tier_quants="$8" tier_fractions="$9" disable_gateup="${10}" actual_avg_sum_bits="${11}" status="${12}"
  local wikitext2="${13}" c4="${14}" calibration_seconds="${15}" duquant_seconds="${16}"
  local inference_wikitext2_seconds="${17}" inference_c4_seconds="${18}" plan_path="${19}" output_dir="${20}"
  local plan_log="${21}" run_log="${22}"

  {
    csv_quote "$label"; printf ','
    csv_quote "$ose"; printf ','
    csv_quote "$allocation"; printf ','
    csv_quote "$planner_metric"; printf ','
    csv_quote "$topk"; printf ','
    csv_quote "$outlier_quant"; printf ','
    csv_quote "$outlier_score"; printf ','
    csv_quote "$tier_quants"; printf ','
    csv_quote "$tier_fractions"; printf ','
    csv_quote "$disable_gateup"; printf ','
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
    csv_quote "$plan_path"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$plan_log"; printf ','
    csv_quote "$run_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

total="${#CASES[@]}"
start_ts="$(date +%s)"

echo "[component-routed-tail-2048] output root: $OUT_ROOT"
echo "[component-routed-tail-2048] test_dataset: $TEST_DATASET"
echo "[component-routed-tail-2048] tasks: ${TASKS:-<none>}"
echo "[component-routed-tail-2048] final metric: routed_tail"
echo "[component-routed-tail-2048] final tier: $FINAL_TIER_QUANTS / $FINAL_TIER_FRACTIONS"
echo "[component-routed-tail-2048] weight_channel_group_size: $WEIGHT_CHANNEL_GROUP_SIZE"

for idx in "${!CASES[@]}"; do
  IFS='|' read -r label ose allocation planner_metric topk outlier_quant outlier_score tier_quants tier_fractions disable_gateup <<< "${CASES[$idx]}"
  case_num=$((idx + 1))
  case_dir="$OUT_ROOT/$label"
  plan_dir="$case_dir/plans"
  output_dir="$case_dir/output"
  plan_log="$case_dir/plan.log"
  run_log="$case_dir/run.log"
  plan_path="$plan_dir/moe_quant_plan_${planner_metric}.json"
  summary_path="$plan_dir/moe_quant_plan_${planner_metric}_summary.json"

  mkdir -p "$plan_dir" "$output_dir"
  echo
  echo "[component-routed-tail-2048] case $case_num/$total: $label"

  status="ok"
  if ! run_logged "$plan_log" \
    python3 tools/olmoe_expert_distribution_planner.py \
      --model "$MODEL" \
      --model_name olmoe \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$plan_dir" \
      --metrics "$planner_metric" \
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
      --route_alpha 1.0 \
      --score_transform raw \
      --input_score_coef 1.0 \
      --down_score_coef 1.0 \
      --weight_score_coef 1.0 \
      --weight_score_projs gate_proj,up_proj,down_proj \
      --moe_outlier_topk "$topk" \
      --moe_outlier_score "$outlier_score" \
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
          MOE_OUTLIER_TOPK="$topk" \
          MOE_OUTLIER_QUANT="$outlier_quant" \
          MOE_OUTLIER_SCORE="$outlier_score" \
          DISABLE_MOE_GATE_UP_DUQUANT_ROTATION="$disable_gateup" \
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

  append_summary "$label" "$ose" "$allocation" "$planner_metric" "$topk" "$outlier_quant" "$outlier_score" \
    "$tier_quants" "$tier_fractions" "$disable_gateup" "$actual_avg_sum_bits" "$status" \
    "$wikitext2" "$c4" "$calibration_seconds" "$duquant_seconds" \
    "$inference_wikitext2_seconds" "$inference_c4_seconds" "$plan_path" "$output_dir" "$plan_log" "$run_log"

  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  remaining_cases=$((total - case_num))
  est_remaining=$((remaining_cases * EST_SECONDS_PER_RUN))
  eta="$(date -d "@$((now_ts + est_remaining))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  echo "[component-routed-tail-2048] finished $label with status=$status"
  echo "[component-routed-tail-2048] elapsed=${elapsed}s, estimated_remaining=${est_remaining}s, eta=${eta:-unknown}"
  echo "[component-routed-tail-2048] summary: $SUMMARY_CSV"
done

echo
echo "[component-routed-tail-2048] all done: $SUMMARY_CSV"
