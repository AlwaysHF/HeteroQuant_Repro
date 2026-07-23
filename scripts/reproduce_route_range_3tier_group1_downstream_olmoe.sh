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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/reproduce_route_range_3tier_group1_downstream_olmoe_$(date +%Y%m%d_%H%M%S)}"

TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"

NSAMPLES_PLAN="${NSAMPLES_PLAN:-32}"
NSAMPLES_RUN="${NSAMPLES_RUN:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"

TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
TIER_FRACTIONS="${TIER_FRACTIONS:-0.234375,0.59375,0.171875}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-1}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-1}"

MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-w8a8}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"

FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/home/lwk/EAQuant_QwenMoE/cache}"

mkdir -p "$CACHE_DIR" "$OUT_ROOT"

planner_cache_name="planner_dataloader_olmoe_wikitext2_${NSAMPLES_PLAN}_${SEQ_LENGTH}_2.cache"
if [[ ! -f "$CACHE_DIR/$planner_cache_name" && -f "$FALLBACK_CACHE_DIR/$planner_cache_name" ]]; then
  echo "[route-range-3tier-g1] copy cached planner dataloader from $FALLBACK_CACHE_DIR/$planner_cache_name"
  cp -a "$FALLBACK_CACHE_DIR/$planner_cache_name" "$CACHE_DIR/"
fi

PLAN_DIR="$OUT_ROOT/plans"
OUTPUT_DIR="$OUT_ROOT/output"
PLAN_LOG="$OUT_ROOT/plan.log"
RUN_LOG="$OUT_ROOT/run.log"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
PLAN_PATH="$PLAN_DIR/moe_quant_plan_routed_range.json"
PLAN_SUMMARY="$PLAN_DIR/moe_quant_plan_routed_range_summary.json"

mkdir -p "$PLAN_DIR" "$OUTPUT_DIR"

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

extract_last_value() {
  local pattern="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    grep -E "$pattern" "$file" | tail -n 1 | sed -E 's/.*: *([0-9.]+).*/\1/' || true
  fi
}

extract_task_value() {
  local task="$1"
  local file="$2"
  python3 - "$task" "$file" <<'PY'
import ast
import sys

task, path = sys.argv[1], sys.argv[2]
lines = open(path, errors="ignore").read().splitlines()
payload = None
for idx, line in enumerate(lines):
    if "final t_results" in line and idx + 1 < len(lines):
        payload = lines[idx + 1].split("INFO ", 1)[-1].strip()
if payload is None:
    for line in reversed(lines):
        if "(utils.py 314)" in line and "INFO {" in line:
            payload = line.split("INFO ", 1)[-1].strip()
            break
if payload is None:
    print("")
    raise SystemExit
data = ast.literal_eval(payload)
value = data.get(task, "")
if isinstance(value, dict):
    value = next(iter(value.values()))
print(value)
PY
}

extract_acc_avg() {
  local file="$1"
  python3 - "$file" <<'PY'
import ast
import sys

path = sys.argv[1]
lines = open(path, errors="ignore").read().splitlines()
payload = None
for idx, line in enumerate(lines):
    if "final t_results" in line and idx + 1 < len(lines):
        payload = lines[idx + 1].split("INFO ", 1)[-1].strip()
if payload is None:
    for line in reversed(lines):
        if "(utils.py 314)" in line and "INFO {" in line:
            payload = line.split("INFO ", 1)[-1].strip()
            break
if payload is None:
    print("")
    raise SystemExit
print(ast.literal_eval(payload).get("acc_avg", ""))
PY
}

echo "[route-range-3tier-g1] output root: $OUT_ROOT"
echo "[route-range-3tier-g1] formula: route_range = (input_range + down_range + weight_range) * route_norm"
echo "[route-range-3tier-g1] tier_quants: $TIER_QUANTS"
echo "[route-range-3tier-g1] tier_fractions: $TIER_FRACTIONS"
echo "[route-range-3tier-g1] weight_channel_group_size: $WEIGHT_CHANNEL_GROUP_SIZE"
echo "[route-range-3tier-g1] tasks: $TASKS"

status="ok"

if ! run_logged "$PLAN_LOG" \
  python3 tools/olmoe_expert_distribution_planner.py \
    --model "$MODEL" \
    --model_name olmoe \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$PLAN_DIR" \
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
    --moe_outlier_topk "$MOE_OUTLIER_TOPK" \
    --moe_outlier_score "$MOE_OUTLIER_SCORE" \
    --smooth \
    --fc1_scale_merge act_p99 \
    --act_mean_beta 1 \
    --reuse_dataloader_cache; then
  status="plan_failed"
fi

if [[ "$status" == "ok" ]]; then
  if ! run_logged "$RUN_LOG" \
    env PLAN_PATH="$PLAN_PATH" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        MODEL="$MODEL" \
        CACHE_DIR="$CACHE_DIR" \
        TEST_DATASET="$TEST_DATASET" \
        TASKS="$TASKS" \
        NUM_FEWSHOT="$NUM_FEWSHOT" \
        BATCH_SIZE="$BATCH_SIZE" \
        NSAMPLES="$NSAMPLES_RUN" \
        SEQ_LENGTH="$SEQ_LENGTH" \
        SCALE_SEARCH_STEPS="$SCALE_SEARCH_STEPS" \
        FAST_MOE_CALIB_TOKENS="$FAST_MOE_CALIB_TOKENS" \
        WEIGHT_CHANNEL_GROUP_SIZE="$WEIGHT_CHANNEL_GROUP_SIZE" \
        ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
        MOE_OUTLIER_TOPK="$MOE_OUTLIER_TOPK" \
        MOE_OUTLIER_QUANT="$MOE_OUTLIER_QUANT" \
        MOE_OUTLIER_SCORE="$MOE_OUTLIER_SCORE" \
        DISABLE_MOE_GATE_UP_DUQUANT_ROTATION=1 \
        bash scripts/reproduce_olmoe_691.sh; then
    status="run_failed"
  fi
fi

rank_log="$(find "$OUTPUT_DIR" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
actual_avg_sum_bits=""
if [[ -f "$PLAN_SUMMARY" ]]; then
  actual_avg_sum_bits="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("actual_avg_sum_bits",""))' "$PLAN_SUMMARY")"
fi

wikitext2="$(extract_last_value 'wikitext2 :' "$rank_log")"
c4="$(extract_last_value 'c4 :' "$rank_log")"
calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"
task_all_seconds="$(extract_last_value '\[timing\] inference_seconds\.task\.all:' "$rank_log")"

acc_avg=""
arc_challenge=""
arc_easy=""
boolq=""
openbookqa=""
piqa=""
winogrande=""
if [[ -f "$rank_log" ]]; then
  acc_avg="$(extract_acc_avg "$rank_log")"
  arc_challenge="$(extract_task_value arc_challenge "$rank_log")"
  arc_easy="$(extract_task_value arc_easy "$rank_log")"
  boolq="$(extract_task_value boolq "$rank_log")"
  openbookqa="$(extract_task_value openbookqa "$rank_log")"
  piqa="$(extract_task_value piqa "$rank_log")"
  winogrande="$(extract_task_value winogrande "$rank_log")"
fi

printf 'label,metric,tier_quants,tier_fractions,weight_channel_group_size,router_weight_channel_group_size,moe_outlier_topk,moe_outlier_quant,moe_outlier_score,actual_avg_sum_bits,status,wikitext2,c4,acc_avg,arc_challenge,arc_easy,boolq,openbookqa,piqa,winogrande,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,inference_task_all_seconds,plan_path,output_dir,plan_log,run_log,rank_log\n' > "$SUMMARY_CSV"
{
  csv_quote "3tier_w4a4_w5a8_w6a8"; printf ','
  csv_quote "route_range"; printf ','
  csv_quote "$TIER_QUANTS"; printf ','
  csv_quote "$TIER_FRACTIONS"; printf ','
  csv_quote "$WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
  csv_quote "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
  csv_quote "$MOE_OUTLIER_TOPK"; printf ','
  csv_quote "$MOE_OUTLIER_QUANT"; printf ','
  csv_quote "$MOE_OUTLIER_SCORE"; printf ','
  csv_quote "$actual_avg_sum_bits"; printf ','
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
  csv_quote "$calibration_seconds"; printf ','
  csv_quote "$duquant_seconds"; printf ','
  csv_quote "$inference_wikitext2_seconds"; printf ','
  csv_quote "$inference_c4_seconds"; printf ','
  csv_quote "$task_all_seconds"; printf ','
  csv_quote "$PLAN_PATH"; printf ','
  csv_quote "$OUTPUT_DIR"; printf ','
  csv_quote "$PLAN_LOG"; printf ','
  csv_quote "$RUN_LOG"; printf ','
  csv_quote "$rank_log"; printf '\n'
} >> "$SUMMARY_CSV"

echo "[route-range-3tier-g1] done with status=$status"
echo "[route-range-3tier-g1] summary: $SUMMARY_CSV"
