#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/final_routed_tail_downstream_group2048_${RUN_ID}}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-2048}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"
EST_SECONDS_PER_MODEL="${EST_SECONDS_PER_MODEL:-14400}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-w8a8}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"
TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
TIER_FRACTIONS="${TIER_FRACTIONS:-0.234375,0.59375,0.171875}"

OLMOE_MODEL="${OLMOE_MODEL:-$ROOT/local_models/olmoe_compat}"
QWEN_MODEL="${QWEN_MODEL:-$ROOT/local_models/Qwen1.5-MoE-A2.7B}"

mkdir -p "$CACHE_DIR" "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
printf 'label,model_name,model_path,metric,tier_quants,tier_fractions,moe_outlier_topk,moe_outlier_quant,moe_outlier_score,weight_channel_group_size,router_weight_channel_group_size,status,wikitext2,c4,acc_avg,arc_challenge,arc_easy,boolq,openbookqa,piqa,winogrande,plan_path,output_dir,plan_log,run_log\n' > "$SUMMARY_CSV"

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

planner_cache_name_for() {
  local model_name="$1"
  printf 'planner_dataloader_%s_wikitext2_%s_%s_2.cache' "$model_name" "$NSAMPLES_PLAN" "$SEQ_LENGTH"
}

copy_planner_cache_if_available() {
  local model_name="$1"
  local planner_cache_name
  planner_cache_name="$(planner_cache_name_for "$model_name")"
  if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
    echo "[final-downstream] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
    cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
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
    if "INFO {" not in line or ("acc_avg" not in line and "wikitext2" not in line and "c4" not in line):
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
  local model_name="$2"
  local model_path="$3"
  local status="$4"
  local plan_path="$5"
  local output_dir="$6"
  local plan_log="$7"
  local run_log="$8"
  local values
  values="$(extract_results_tsv "$run_log")"
  IFS=$'\t' read -r wikitext2 c4 acc_avg arc_challenge arc_easy boolq openbookqa piqa winogrande <<< "$values"
  {
    csv_quote "$label"; printf ','
    csv_quote "$model_name"; printf ','
    csv_quote "$model_path"; printf ','
    csv_quote "routed_tail"; printf ','
    csv_quote "$TIER_QUANTS"; printf ','
    csv_quote "$TIER_FRACTIONS"; printf ','
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
    csv_quote "$plan_path"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$plan_log"; printf ','
    csv_quote "$run_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

run_one_model() {
  local label="$1"
  local model_name="$2"
  local model_path="$3"
  local reproduce_script="$4"
  local case_dir="$OUT_ROOT/$label"
  local plan_dir="$case_dir/plans"
  local output_dir="$case_dir/output"
  local plan_log="$case_dir/plan.log"
  local run_log="$case_dir/run.log"
  local plan_path="$plan_dir/moe_quant_plan_routed_tail.json"

  mkdir -p "$plan_dir" "$output_dir"
  copy_planner_cache_if_available "$model_name"

  echo
  echo "[final-downstream] model: $label"
  echo "[final-downstream] plan: $plan_path"
  echo "[final-downstream] run log: $run_log"

  local status="ok"
  if ! run_logged "$plan_log" \
    python3 tools/olmoe_expert_distribution_planner.py \
      --model "$model_path" \
      --model_name "$model_name" \
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
          MODEL="$model_path" \
          CACHE_DIR="$CACHE_DIR" \
          TEST_DATASET="$TEST_DATASET" \
          TASKS="$TASKS" \
          NUM_FEWSHOT="$NUM_FEWSHOT" \
          BATCH_SIZE="$BATCH_SIZE" \
          NSAMPLES="$NSAMPLES_RUN" \
          SEQ_LENGTH="$SEQ_LENGTH" \
          SCALE_SEARCH_STEPS="$SCALE_SEARCH_STEPS" \
          FAST_MOE_CALIB_TOKENS="$FAST_MOE_CALIB_TOKENS" \
          MOE_OUTLIER_TOPK="$MOE_OUTLIER_TOPK" \
          MOE_OUTLIER_QUANT="$MOE_OUTLIER_QUANT" \
          MOE_OUTLIER_SCORE="$MOE_OUTLIER_SCORE" \
          WEIGHT_CHANNEL_GROUP_SIZE="$WEIGHT_CHANNEL_GROUP_SIZE" \
          ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
          bash "$reproduce_script"; then
      status="run_failed"
    fi
  fi

  append_summary "$label" "$model_name" "$model_path" "$status" "$plan_path" "$output_dir" "$plan_log" "$run_log"
  echo "[final-downstream] finished $label with status=$status"
  echo "[final-downstream] summary: $SUMMARY_CSV"
}

start_ts="$(date +%s)"
echo "[final-downstream] output root: $OUT_ROOT"
echo "[final-downstream] tasks: $TASKS"
echo "[final-downstream] test_dataset: $TEST_DATASET"
echo "[final-downstream] group size: $WEIGHT_CHANNEL_GROUP_SIZE"
echo "[final-downstream] metric/tier/OSE: routed_tail / $TIER_QUANTS / topk=$MOE_OUTLIER_TOPK $MOE_OUTLIER_QUANT $MOE_OUTLIER_SCORE"

run_one_model "olmoe_final_routed_tail_group2048" "olmoe" "$OLMOE_MODEL" "scripts/reproduce_olmoe_691.sh"

now_ts="$(date +%s)"
eta="$(date -d "@$((now_ts + EST_SECONDS_PER_MODEL))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
echo "[final-downstream] OLMoE done, rough ETA after Qwen: ${eta:-unknown}"

run_one_model "qwen_final_routed_tail_group2048" "qwen2_moe" "$QWEN_MODEL" "scripts/reproduce_qwen_744.sh"

end_ts="$(date +%s)"
echo
echo "[final-downstream] all done in $((end_ts - start_ts))s"
echo "[final-downstream] summary: $SUMMARY_CSV"
