#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/route_range_aw11_g1_g2048_olmoe_qwen_ppl_downstream_${RUN_ID}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

OLMOE_MODEL="${OLMOE_MODEL:-$ROOT/local_models/olmoe_compat}"
QWEN_MODEL="${QWEN_MODEL:-$ROOT/local_models/Qwen1.5-MoE-A2.7B}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

CALIB_DATASET="${CALIB_DATASET:-wikitext2}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"

AW_LABEL="${AW_LABEL:-aw11}"
TARGET_AVG_SUM_BITS="${TARGET_AVG_SUM_BITS:-11.0}"
TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
# A4W4/A8W5/A8W6 have total bits 8/13/14. 27/64,30/64,7/64 gives exactly 11.0.
TIER_FRACTIONS="${TIER_FRACTIONS:-0.421875,0.46875,0.109375}"

BASE_W_BITS="${BASE_W_BITS:-4}"
BASE_A_BITS="${BASE_A_BITS:-8}"
ROUTER_W_BITS="${ROUTER_W_BITS:-8}"
ROUTER_A_BITS="${ROUTER_A_BITS:-8}"
GROUP_SIZES="${GROUP_SIZES:-1,2048}"

MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-w8a8}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"

mkdir -p "$OUT_ROOT" "$CACHE_DIR"

csv_quote() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

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
    grep -E "$pattern" "$file" | tail -n 1 | sed -E 's/.*: *([0-9.eE+-]+).*/\1/' || true
  fi
}

extract_results_tsv() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(errors="ignore") if path.exists() else ""
records = []
for line in text.splitlines():
    if "INFO {" not in line or ("wikitext2" not in line and "c4" not in line and "acc_avg" not in line):
        continue
    payload = line.split("INFO ", 1)[-1]
    try:
        obj = ast.literal_eval(payload)
    except Exception:
        continue
    if isinstance(obj, dict):
        records.append(obj)

result = {}
for obj in records:
    if "wikitext2" in obj or "c4" in obj:
        result.update(obj)
for obj in records:
    if "acc_avg" in obj:
        result.update(obj)

def task_value(name):
    value = result.get(name, "")
    if isinstance(value, dict):
        if "acc_norm" in value:
            return value["acc_norm"]
        if "acc" in value:
            return value["acc"]
        if value:
            return next(iter(value.values()))
    return value

fields = [
    result.get("wikitext2", ""),
    result.get("c4", ""),
    result.get("acc_avg", ""),
    task_value("arc_challenge"),
    task_value("arc_easy"),
    task_value("boolq"),
    task_value("openbookqa"),
    task_value("piqa"),
    task_value("winogrande"),
]
print("\t".join("" if value is None else str(value) for value in fields))
PY
}

copy_planner_cache_if_available() {
  local model_name="$1"
  local cache_name="planner_dataloader_${model_name}_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
  if [[ ! -f "$CACHE_DIR/$cache_name" && -f "$FALLBACK_CACHE_DIR/$cache_name" ]]; then
    echo "[route-range-aw11] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$cache_name"
    cp -a "$FALLBACK_CACHE_DIR/$cache_name" "$CACHE_DIR/"
  fi
}

append_summary() {
  local label="$1" model_name="$2" model_path="$3" group_size="$4" status="$5"
  local actual_avg_sum_bits="$6" plan_path="$7" output_dir="$8" plan_log="$9" run_log="${10}" rank_log="${11}"

  local values wikitext2 c4 acc_avg arc_challenge arc_easy boolq openbookqa piqa winogrande
  local calibration_seconds smooth_seconds duquant_seconds inference_wikitext2_seconds inference_c4_seconds inference_task_seconds
  values="$(extract_results_tsv "$rank_log")"
  IFS=$'\t' read -r wikitext2 c4 acc_avg arc_challenge arc_easy boolq openbookqa piqa winogrande <<< "$values"
  calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
  smooth_seconds="$(extract_last_value '\[timing\] smooth_seconds:' "$rank_log")"
  duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
  inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
  inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"
  inference_task_seconds="$(extract_last_value '\[timing\] inference_seconds\.task\.all:' "$rank_log")"

  {
    csv_quote "$label"; printf ','
    csv_quote "$model_name"; printf ','
    csv_quote "$model_path"; printf ','
    csv_quote "routed_range"; printf ','
    csv_quote "$TARGET_AVG_SUM_BITS"; printf ','
    csv_quote "$TIER_QUANTS"; printf ','
    csv_quote "$TIER_FRACTIONS"; printf ','
    csv_quote "$BASE_W_BITS"; printf ','
    csv_quote "$BASE_A_BITS"; printf ','
    csv_quote "$ROUTER_W_BITS"; printf ','
    csv_quote "$ROUTER_A_BITS"; printf ','
    csv_quote "$group_size"; printf ','
    csv_quote "$group_size"; printf ','
    csv_quote "$MOE_OUTLIER_TOPK"; printf ','
    csv_quote "$MOE_OUTLIER_QUANT"; printf ','
    csv_quote "$MOE_OUTLIER_SCORE"; printf ','
    csv_quote "$status"; printf ','
    csv_quote "$wikitext2"; printf ','
    csv_quote "$c4"; printf ','
    csv_quote "$acc_avg"; printf ','
    csv_quote "$arc_challenge"; printf ','
    csv_quote "$arc_easy"; printf ','
    csv_quote "$boolq"; printf ','
    csv_quote "$openbookqa"; printf ','
    csv_quote "$piqa"; printf ','
    csv_quote "$winogrande"; printf ','
    csv_quote "$actual_avg_sum_bits"; printf ','
    csv_quote "$calibration_seconds"; printf ','
    csv_quote "$smooth_seconds"; printf ','
    csv_quote "$duquant_seconds"; printf ','
    csv_quote "$inference_wikitext2_seconds"; printf ','
    csv_quote "$inference_c4_seconds"; printf ','
    csv_quote "$inference_task_seconds"; printf ','
    csv_quote "$plan_path"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$plan_log"; printf ','
    csv_quote "$run_log"; printf ','
    csv_quote "$rank_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

printf 'label,model_name,model_path,metric,target_avg_sum_bits,tier_quants,tier_fractions,base_wbits,base_abits,router_wbits,router_abits,weight_channel_group_size,router_weight_channel_group_size,moe_outlier_topk,moe_outlier_quant,moe_outlier_score,status,wikitext2,c4,acc_avg,arc_challenge,arc_easy,boolq,openbookqa,piqa,winogrande,actual_avg_sum_bits,calibration_seconds,smooth_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,inference_task_seconds,plan_path,output_dir,plan_log,run_log,rank_log\n' > "$SUMMARY_CSV"

run_one() {
  local model_name="$1"
  local model_path="$2"
  local group_size="$3"
  local label="${model_name}_route_range_${AW_LABEL}_g${group_size}_ppl_downstream"
  local case_dir="$OUT_ROOT/$label"
  local plan_dir="$case_dir/plans"
  local output_dir="$case_dir/output"
  local plan_log="$case_dir/plan.log"
  local run_log="$case_dir/run.log"
  local plan_path="$plan_dir/moe_quant_plan_routed_range.json"
  local summary_path="$plan_dir/moe_quant_plan_routed_range_summary.json"

  mkdir -p "$plan_dir" "$output_dir"
  copy_planner_cache_if_available "$model_name"

  echo
  echo "[route-range-aw11] start $label"
  echo "[route-range-aw11] model=$model_path"
  echo "[route-range-aw11] group_size=$group_size target_avg_sum_bits=$TARGET_AVG_SUM_BITS"
  echo "[route-range-aw11] tier_quants=$TIER_QUANTS"
  echo "[route-range-aw11] tier_fractions=$TIER_FRACTIONS"

  local status="ok"
  if ! run_logged "$plan_log" \
    python3 tools/olmoe_expert_distribution_planner.py \
      --model "$model_path" \
      --model_name "$model_name" \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$plan_dir" \
      --metrics routed_range \
      --calib_dataset "$CALIB_DATASET" \
      --nsamples "$NSAMPLES_PLAN" \
      --seq_length "$SEQ_LENGTH" \
      --seed 2 \
      --attn_implementation eager \
      --selection_scope per_layer \
      --plan_granularity expert \
      --target_avg_sum_bits "$TARGET_AVG_SUM_BITS" \
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
      python3 main.py \
        --model "$model_path" \
        --model_name "$model_name" \
        --calib_dataset "$CALIB_DATASET" \
        --wbits "$BASE_W_BITS" --abits "$BASE_A_BITS" \
        --router_wbits "$ROUTER_W_BITS" --router_abits "$ROUTER_A_BITS" \
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
        --tasks "$TASKS" \
        --num_fewshot "$NUM_FEWSHOT" \
        --batch_size "$BATCH_SIZE" \
        --group_size -1 \
        --weight_channel_group_size "$group_size" \
        --router_weight_channel_group_size "$group_size" \
        --act_group_size -1 \
        --smooth \
        --fc1_scale_merge act_p99 \
        --act_mean_beta 1 \
        --moe_down_smooth_mode otsu \
        --moe_outlier_topk "$MOE_OUTLIER_TOPK" \
        --moe_outlier_quant "$MOE_OUTLIER_QUANT" \
        --moe_outlier_score "$MOE_OUTLIER_SCORE" \
        --disable_moe_gate_up_duquant_rotation \
        --moe_quant_plan "$plan_path" \
        --fast_moe_down_calibration \
        --fast_moe_calib_tokens "$FAST_MOE_CALIB_TOKENS"; then
      status="run_failed"
    fi
  fi

  local actual_avg_sum_bits=""
  if [[ -f "$summary_path" ]]; then
    actual_avg_sum_bits="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("actual_avg_sum_bits",""))' "$summary_path")"
  fi
  local rank_log
  rank_log="$(find "$output_dir" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
  append_summary "$label" "$model_name" "$model_path" "$group_size" "$status" "$actual_avg_sum_bits" \
    "$plan_path" "$output_dir" "$plan_log" "$run_log" "$rank_log"
  echo "[route-range-aw11] finished $label status=$status"
  echo "[route-range-aw11] summary: $SUMMARY_CSV"
}

IFS=',' read -r -a group_size_list <<< "$GROUP_SIZES"

echo "[route-range-aw11] output root: $OUT_ROOT"
echo "[route-range-aw11] metric: routed_range = route * (input_range + down_range + weight_range)"
echo "[route-range-aw11] groups: $GROUP_SIZES"
echo "[route-range-aw11] PPL: $TEST_DATASET"
echo "[route-range-aw11] downstream: $TASKS"
echo "[route-range-aw11] candidate tiers: $TIER_QUANTS / $TIER_FRACTIONS"

for group_size in "${group_size_list[@]}"; do
  run_one "olmoe" "$OLMOE_MODEL" "$group_size"
  run_one "qwen2_moe" "$QWEN_MODEL" "$group_size"
done

echo
echo "[route-range-aw11] all done"
echo "[route-range-aw11] summary: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
