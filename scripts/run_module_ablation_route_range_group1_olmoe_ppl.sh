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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/module_ablation_route_range_group1_olmoe_ppl_${RUN_ID}}"

TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-1800}"

WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-1}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"

# 6.9024 reference uses the old layer-shared smooth scale OSE selection.
# If you want the newer per-expert activation-scale selection, set OSE_SCORE=shared_scale.
OSE_SCORE="${OSE_SCORE:-smooth_scale}"
OSE_TOPK="${OSE_TOPK:-64}"
OSE_QUANT="${OSE_QUANT:-w8a8}"

UNIFORM_LABEL="${UNIFORM_LABEL:-uniform_w4a8}"
FINAL_TIER_QUANTS="${FINAL_TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
FINAL_TIER_FRACTIONS="${FINAL_TIER_FRACTIONS:-0.234375,0.59375,0.171875}"

FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

mkdir -p "$CACHE_DIR" "$OUT_ROOT"

planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[module-ablation] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'label,migration,ose,tier,ose_mode,planner_metric,moe_outlier_topk,moe_outlier_quant,moe_outlier_score,tier_quants,tier_fractions,weight_channel_group_size,router_weight_channel_group_size,actual_avg_sum_bits,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,plan_path,output_dir,plan_log,run_log,rank_log\n' > "$SUMMARY_CSV"

CASES=(
  "no_migration_no_ose_no_tier|0|0|none|none|none|0|same|weight_max||"
  "migration_no_ose_no_tier|1|0|none|none|none|0|same|$OSE_SCORE||"
  "migration_random_ose_no_tier|1|1|none|random|none|$OSE_TOPK|$OSE_QUANT|random||"
  "migration_ose_no_tier|1|1|none|selected|none|$OSE_TOPK|$OSE_QUANT|$OSE_SCORE||"
  "migration_ose_random_tier|1|1|random|selected|random|$OSE_TOPK|$OSE_QUANT|$OSE_SCORE|$FINAL_TIER_QUANTS|$FINAL_TIER_FRACTIONS"
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
  local label="$1" migration="$2" ose="$3" tier="$4" ose_mode="$5" planner_metric="$6"
  local topk="$7" outlier_quant="$8" outlier_score="$9" tier_quants="${10}" tier_fractions="${11}"
  local actual_avg_sum_bits="${12}" status="${13}" wikitext2="${14}" c4="${15}"
  local calibration_seconds="${16}" duquant_seconds="${17}" inference_wikitext2_seconds="${18}"
  local inference_c4_seconds="${19}" plan_path="${20}" output_dir="${21}" plan_log="${22}" run_log="${23}" rank_log="${24}"

  {
    csv_quote "$label"; printf ','
    csv_quote "$migration"; printf ','
    csv_quote "$ose"; printf ','
    csv_quote "$tier"; printf ','
    csv_quote "$ose_mode"; printf ','
    csv_quote "$planner_metric"; printf ','
    csv_quote "$topk"; printf ','
    csv_quote "$outlier_quant"; printf ','
    csv_quote "$outlier_score"; printf ','
    csv_quote "${tier_quants:-$UNIFORM_LABEL}"; printf ','
    csv_quote "${tier_fractions:-1.0}"; printf ','
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
    csv_quote "$run_log"; printf ','
    csv_quote "$rank_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

generate_plan_if_needed() {
  local planner_metric="$1" topk="$2" outlier_score="$3" tier_quants="$4" tier_fractions="$5" plan_dir="$6" plan_log="$7"
  if [[ "$planner_metric" == "none" ]]; then
    return 0
  fi

  mkdir -p "$plan_dir"
  run_logged "$plan_log" \
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
      --weight_channel_group_size "$WEIGHT_CHANNEL_GROUP_SIZE" \
      --moe_outlier_topk "$topk" \
      --moe_outlier_score "$outlier_score" \
      --smooth \
      --fc1_scale_merge act_p99 \
      --act_mean_beta 1 \
      --reuse_dataloader_cache
}

run_case() {
  local migration="$1" topk="$2" outlier_quant="$3" outlier_score="$4" plan_path="$5" output_dir="$6" run_log="$7"

  local smooth_args=()
  if [[ "$migration" == "1" ]]; then
    smooth_args=(
      --smooth
      --fc1_scale_merge act_p99
      --act_mean_beta 1
      --moe_down_smooth_mode otsu
    )
  fi

  local plan_args=()
  if [[ -n "$plan_path" ]]; then
    plan_args=(--moe_quant_plan "$plan_path")
  fi

  run_logged "$run_log" \
    python3 main.py \
      --model "$MODEL" \
      --model_name olmoe \
      --wbits 4 --abits 8 \
      --router_wbits 8 --router_abits 8 \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$output_dir" \
      --w_dynamic_method per_channel_tensor \
      --a_dynamic_method per_token \
      --router_w_dynamic_method per_channel_kl_top0 \
      --nsamples "$NSAMPLES_RUN" \
      --seq_length "$SEQ_LENGTH" \
      --scale_search_steps "$SCALE_SEARCH_STEPS" \
      --eval_ppl \
      --test_dataset "$TEST_DATASET" \
      --group_size -1 \
      --weight_channel_group_size "$WEIGHT_CHANNEL_GROUP_SIZE" \
      --router_weight_channel_group_size "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
      --act_group_size -1 \
      "${smooth_args[@]}" \
      --moe_outlier_topk "$topk" \
      --moe_outlier_quant "$outlier_quant" \
      --moe_outlier_score "$outlier_score" \
      --disable_moe_gate_up_duquant_rotation \
      "${plan_args[@]}" \
      --fast_moe_down_calibration \
      --fast_moe_calib_tokens "$FAST_MOE_CALIB_TOKENS"
}

total="${#CASES[@]}"
start_ts="$(date +%s)"

echo "[module-ablation] output root: $OUT_ROOT"
echo "[module-ablation] reference full method is not rerun; expected old layer-shared OSE WikiText2 PPL around 6.9024"
echo "[module-ablation] model: $MODEL"
echo "[module-ablation] datasets: $TEST_DATASET"
echo "[module-ablation] group size: $WEIGHT_CHANNEL_GROUP_SIZE"
echo "[module-ablation] selected OSE score: $OSE_SCORE, topK=$OSE_TOPK, quant=$OSE_QUANT"
echo "[module-ablation] final tier: $FINAL_TIER_QUANTS / $FINAL_TIER_FRACTIONS"

for idx in "${!CASES[@]}"; do
  IFS='|' read -r label migration ose tier ose_mode planner_metric topk outlier_quant outlier_score tier_quants tier_fractions <<< "${CASES[$idx]}"
  case_num=$((idx + 1))
  case_dir="$OUT_ROOT/$label"
  output_dir="$case_dir/output"
  plan_dir="$case_dir/plans"
  plan_log="$case_dir/plan.log"
  run_log="$case_dir/run.log"
  mkdir -p "$case_dir" "$output_dir"

  plan_path=""
  summary_path=""
  actual_avg_sum_bits=""
  status="ok"

  echo
  echo "[module-ablation] case $case_num/$total: $label"
  echo "[module-ablation] migration=$migration ose=$ose tier=$tier ose_mode=$ose_mode metric=$planner_metric score=$outlier_score"

  if [[ "$planner_metric" != "none" ]]; then
    plan_path="$plan_dir/moe_quant_plan_${planner_metric}.json"
    summary_path="$plan_dir/moe_quant_plan_${planner_metric}_summary.json"
    if ! generate_plan_if_needed "$planner_metric" "$topk" "$outlier_score" "$tier_quants" "$tier_fractions" "$plan_dir" "$plan_log"; then
      status="plan_failed"
    fi
    if [[ -f "$summary_path" ]]; then
      actual_avg_sum_bits="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("actual_avg_sum_bits",""))' "$summary_path")"
    fi
  else
    plan_log=""
  fi

  if [[ "$status" == "ok" ]]; then
    if ! run_case "$migration" "$topk" "$outlier_quant" "$outlier_score" "$plan_path" "$output_dir" "$run_log"; then
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

  append_summary "$label" "$migration" "$ose" "$tier" "$ose_mode" "$planner_metric" \
    "$topk" "$outlier_quant" "$outlier_score" "$tier_quants" "$tier_fractions" \
    "$actual_avg_sum_bits" "$status" "$wikitext2" "$c4" \
    "$calibration_seconds" "$duquant_seconds" \
    "$inference_wikitext2_seconds" "$inference_c4_seconds" \
    "$plan_path" "$output_dir" "$plan_log" "$run_log" "$rank_log"

  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  remaining=$((total - case_num))
  est_remaining=$((remaining * EST_SECONDS_PER_RUN))
  eta="$(date -d "@$((now_ts + est_remaining))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  echo "[module-ablation] finished $label status=$status"
  echo "[module-ablation] elapsed=${elapsed}s estimated_remaining=${est_remaining}s eta=${eta:-unknown}"
  echo "[module-ablation] summary: $SUMMARY_CSV"
done

echo
echo "[module-ablation] all done: $SUMMARY_CSV"
