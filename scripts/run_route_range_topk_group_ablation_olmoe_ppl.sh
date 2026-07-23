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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/route_range_topk_group_ablation_olmoe_ppl_${RUN_ID}}"

TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"

GROUP_SIZES="${GROUP_SIZES:-1,16,32,64,128,256,512,1024,2048}"
TOPKS="${TOPKS:-0,16,32,64,128,256}"

TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
TIER_FRACTIONS="${TIER_FRACTIONS:-0.234375,0.59375,0.171875}"

MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-w8a8}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE_MODE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE_MODE:-same_as_weight}"
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-1400}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

mkdir -p "$CACHE_DIR" "$OUT_ROOT"

planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[route-range-topk-group] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'group_size,topk,metric,tier_quants,tier_fractions,moe_outlier_quant,moe_outlier_score,router_weight_channel_group_size,actual_avg_sum_bits,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,plan_path,output_dir,plan_log,run_log,rank_log\n' > "$SUMMARY_CSV"

IFS=',' read -r -a GROUP_LIST <<< "$GROUP_SIZES"
IFS=',' read -r -a TOPK_LIST <<< "$TOPKS"

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
  local topk="$2"
  local router_group_size="$3"
  local actual_avg_sum_bits="$4"
  local status="$5"
  local wikitext2="$6"
  local c4="$7"
  local calibration_seconds="$8"
  local duquant_seconds="$9"
  local inference_wikitext2_seconds="${10}"
  local inference_c4_seconds="${11}"
  local plan_path="${12}"
  local output_dir="${13}"
  local plan_log="${14}"
  local run_log="${15}"
  local rank_log="${16}"

  {
    csv_quote "$group_size"; printf ','
    csv_quote "$topk"; printf ','
    csv_quote "route_range"; printf ','
    csv_quote "$TIER_QUANTS"; printf ','
    csv_quote "$TIER_FRACTIONS"; printf ','
    csv_quote "$MOE_OUTLIER_QUANT"; printf ','
    csv_quote "$MOE_OUTLIER_SCORE"; printf ','
    csv_quote "$router_group_size"; printf ','
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
    csv_quote "$run_log"; printf ','
    csv_quote "$rank_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

total_groups="${#GROUP_LIST[@]}"
total_topks="${#TOPK_LIST[@]}"
total_runs=$((total_groups * total_topks))
completed_runs=0
start_ts="$(date +%s)"

echo "[route-range-topk-group] output root: $OUT_ROOT"
echo "[route-range-topk-group] model: $MODEL"
echo "[route-range-topk-group] metric: route_range / routed_range"
echo "[route-range-topk-group] formula: (input_range + down_range + weight_range) * route_norm"
echo "[route-range-topk-group] tier_quants: $TIER_QUANTS"
echo "[route-range-topk-group] tier_fractions: $TIER_FRACTIONS"
echo "[route-range-topk-group] group_sizes: $GROUP_SIZES"
echo "[route-range-topk-group] topks: $TOPKS"
echo "[route-range-topk-group] OSE quant=$MOE_OUTLIER_QUANT score=$MOE_OUTLIER_SCORE"
echo "[route-range-topk-group] test_dataset: $TEST_DATASET"
echo "[route-range-topk-group] tasks: ${TASKS:-<none>}"
echo "[route-range-topk-group] total runs: $total_runs"

for group_idx in "${!GROUP_LIST[@]}"; do
  group_size="$(echo "${GROUP_LIST[$group_idx]}" | xargs)"
  [[ -n "$group_size" ]] || continue

  if [[ "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE_MODE" == "same_as_weight" ]]; then
    router_group_size="$group_size"
  else
    router_group_size="$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE_MODE"
  fi

  group_dir="$OUT_ROOT/group_${group_size}"
  plan_dir="$group_dir/plans"
  plan_log="$group_dir/plan.log"
  plan_path="$plan_dir/moe_quant_plan_routed_range.json"
  summary_path="$plan_dir/moe_quant_plan_routed_range_summary.json"
  mkdir -p "$plan_dir"

  echo
  echo "[route-range-topk-group] planning group_size=$group_size router_group_size=$router_group_size"

  plan_status="ok"
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
      --input_score_coef 1.0 \
      --down_score_coef 1.0 \
      --weight_score_coef 1.0 \
      --weight_score_projs gate_proj,up_proj,down_proj \
      --weight_channel_group_size "$group_size" \
      --moe_outlier_topk 64 \
      --moe_outlier_score "$MOE_OUTLIER_SCORE" \
      --smooth \
      --fc1_scale_merge act_p99 \
      --act_mean_beta 1 \
      --reuse_dataloader_cache; then
    plan_status="plan_failed"
  fi

  actual_avg_sum_bits=""
  if [[ -f "$summary_path" ]]; then
    actual_avg_sum_bits="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("actual_avg_sum_bits",""))' "$summary_path")"
  fi

  for topk_idx in "${!TOPK_LIST[@]}"; do
    topk="$(echo "${TOPK_LIST[$topk_idx]}" | xargs)"
    [[ -n "$topk" ]] || continue
    completed_runs=$((completed_runs + 1))

    label="g${group_size}_k${topk}"
    case_dir="$group_dir/topk_${topk}"
    output_dir="$case_dir/output"
    run_log="$case_dir/run.log"
    mkdir -p "$output_dir"

    echo
    echo "[route-range-topk-group] run $completed_runs/$total_runs: group_size=$group_size topk=$topk"

    status="$plan_status"
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
            WEIGHT_CHANNEL_GROUP_SIZE="$group_size" \
            ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="$router_group_size" \
            MOE_OUTLIER_TOPK="$topk" \
            MOE_OUTLIER_QUANT="$MOE_OUTLIER_QUANT" \
            MOE_OUTLIER_SCORE="$MOE_OUTLIER_SCORE" \
            DISABLE_MOE_GATE_UP_DUQUANT_ROTATION=1 \
            bash scripts/reproduce_olmoe_691.sh; then
        status="run_failed"
      fi
    fi

    rank_log="$(find "$output_dir" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
    wikitext2="$(extract_last_value 'wikitext2 :' "$rank_log")"
    c4="$(extract_last_value 'c4 :' "$rank_log")"
    calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
    duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
    inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
    inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"

    append_summary "$group_size" "$topk" "$router_group_size" "$actual_avg_sum_bits" "$status" \
      "$wikitext2" "$c4" "$calibration_seconds" "$duquant_seconds" \
      "$inference_wikitext2_seconds" "$inference_c4_seconds" \
      "$plan_path" "$output_dir" "$plan_log" "$run_log" "$rank_log"

    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    remaining_runs=$((total_runs - completed_runs))
    est_remaining=$((remaining_runs * EST_SECONDS_PER_RUN))
    eta="$(date -d "@$((now_ts + est_remaining))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
    echo "[route-range-topk-group] finished $label status=$status"
    echo "[route-range-topk-group] elapsed=${elapsed}s estimated_remaining=${est_remaining}s eta=${eta:-unknown}"
    echo "[route-range-topk-group] summary: $SUMMARY_CSV"
  done
done

echo
echo "[route-range-topk-group] all done: $SUMMARY_CSV"
