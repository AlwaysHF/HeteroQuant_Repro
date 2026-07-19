#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

MODEL="${MODEL:-$ROOT/local_models/Qwen1.5-MoE-A2.7B}"
MODEL_NAME="${MODEL_NAME:-qwen2_moe}"
REPRO_SCRIPT="${REPRO_SCRIPT:-scripts/reproduce_qwen_744.sh}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/qwen_aw11_aw13_routed_tail_downstream_group2048_${RUN_ID}}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-2048}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-14400}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-w8a8}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"

mkdir -p "$CACHE_DIR" "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'label,target_avg_sum_bits,tier_quants,tier_fractions,metric,moe_outlier_topk,moe_outlier_quant,moe_outlier_score,weight_channel_group_size,router_weight_channel_group_size,status,wikitext2,c4,acc_avg,arc_challenge,arc_easy,boolq,openbookqa,piqa,winogrande,actual_avg_sum_bits,plan_path,output_dir,plan_log,run_log\n' > "$SUMMARY_CSV"

CASES=(
  "aw11|11|w4g-1-a4g-1,w4g-1-a8g-1|0.25,0.75"
  "aw13|13|w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1|0.0625,0.625,0.3125"
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

csv_quote() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

planner_cache_name="planner_dataloader_${MODEL_NAME}_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[qwen-aw11-aw13] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

extract_results_tsv() {
  local run_log="$1"
  python3 - "$run_log" <<'PY'
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

append_summary() {
  local label="$1"
  local target_avg_sum_bits="$2"
  local tier_quants="$3"
  local tier_fractions="$4"
  local status="$5"
  local actual_avg_sum_bits="$6"
  local plan_path="$7"
  local output_dir="$8"
  local plan_log="$9"
  local run_log="${10}"

  local values
  values="$(extract_results_tsv "$run_log")"
  IFS=$'\t' read -r wikitext2 c4 acc_avg arc_challenge arc_easy boolq openbookqa piqa winogrande <<< "$values"

  {
    csv_quote "$label"; printf ','
    csv_quote "$target_avg_sum_bits"; printf ','
    csv_quote "$tier_quants"; printf ','
    csv_quote "$tier_fractions"; printf ','
    csv_quote "routed_tail"; printf ','
    csv_quote "$MOE_OUTLIER_TOPK"; printf ','
    csv_quote "$MOE_OUTLIER_QUANT"; printf ','
    csv_quote "$MOE_OUTLIER_SCORE"; printf ','
    csv_quote "$WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
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
    csv_quote "$plan_path"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$plan_log"; printf ','
    csv_quote "$run_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

total="${#CASES[@]}"
start_ts="$(date +%s)"

echo "[qwen-aw11-aw13] output root: $OUT_ROOT"
echo "[qwen-aw11-aw13] model: $MODEL"
echo "[qwen-aw11-aw13] group size: $WEIGHT_CHANNEL_GROUP_SIZE"
echo "[qwen-aw11-aw13] tasks: $TASKS"
echo "[qwen-aw11-aw13] OSE: topk=$MOE_OUTLIER_TOPK quant=$MOE_OUTLIER_QUANT score=$MOE_OUTLIER_SCORE"

for idx in "${!CASES[@]}"; do
  IFS='|' read -r label target_avg_sum_bits tier_quants tier_fractions <<< "${CASES[$idx]}"
  case_num=$((idx + 1))
  case_dir="$OUT_ROOT/$label"
  plan_dir="$case_dir/plans"
  output_dir="$case_dir/output"
  plan_log="$case_dir/plan.log"
  run_log="$case_dir/run.log"
  plan_path="$plan_dir/moe_quant_plan_routed_tail.json"
  summary_path="$plan_dir/moe_quant_plan_routed_tail_summary.json"

  mkdir -p "$plan_dir" "$output_dir"

  echo
  echo "[qwen-aw11-aw13] case $case_num/$total: $label target_avg_sum_bits=$target_avg_sum_bits"
  echo "[qwen-aw11-aw13] tier_quants=$tier_quants"
  echo "[qwen-aw11-aw13] tier_fractions=$tier_fractions"

  status="ok"
  if ! run_logged "$plan_log" \
    python3 tools/olmoe_expert_distribution_planner.py \
      --model "$MODEL" \
      --model_name "$MODEL_NAME" \
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
      --target_avg_sum_bits "$target_avg_sum_bits" \
      --tier_quants "$tier_quants" \
      --tier_fractions "$tier_fractions" \
      --route_alpha 1.0 \
      --score_transform raw \
      --input_score_coef 1.0 \
      --down_score_coef 1.0 \
      --weight_score_coef 1.0 \
      --weight_score_projs gate_proj,up_proj,down_proj \
      --weight_channel_group_size "$WEIGHT_CHANNEL_GROUP_SIZE" \
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
          NUM_FEWSHOT="$NUM_FEWSHOT" \
          BATCH_SIZE="$BATCH_SIZE" \
          NSAMPLES="$NSAMPLES_RUN" \
          SEQ_LENGTH="$SEQ_LENGTH" \
          FAST_MOE_CALIB_TOKENS="$FAST_MOE_CALIB_TOKENS" \
          MOE_OUTLIER_TOPK="$MOE_OUTLIER_TOPK" \
          MOE_OUTLIER_QUANT="$MOE_OUTLIER_QUANT" \
          MOE_OUTLIER_SCORE="$MOE_OUTLIER_SCORE" \
          WEIGHT_CHANNEL_GROUP_SIZE="$WEIGHT_CHANNEL_GROUP_SIZE" \
          ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
          bash "$REPRO_SCRIPT"; then
      status="run_failed"
    fi
  fi

  actual_avg_sum_bits=""
  if [[ -f "$summary_path" ]]; then
    actual_avg_sum_bits="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("actual_avg_sum_bits",""))' "$summary_path")"
  fi

  append_summary "$label" "$target_avg_sum_bits" "$tier_quants" "$tier_fractions" \
    "$status" "$actual_avg_sum_bits" "$plan_path" "$output_dir" "$plan_log" "$run_log"

  now_ts="$(date +%s)"
  remaining_cases=$((total - case_num))
  est_remaining=$((remaining_cases * EST_SECONDS_PER_RUN))
  eta="$(date -d "@$((now_ts + est_remaining))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  echo "[qwen-aw11-aw13] finished $label with status=$status"
  echo "[qwen-aw11-aw13] elapsed=$((now_ts - start_ts))s, estimated_remaining=${est_remaining}s, eta=${eta:-unknown}"
  echo "[qwen-aw11-aw13] summary: $SUMMARY_CSV"
done

echo
echo "[qwen-aw11-aw13] all done: $SUMMARY_CSV"
