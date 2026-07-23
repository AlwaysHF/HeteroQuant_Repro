#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

MODEL_NAME="${MODEL_NAME:-olmoe}"
case "${MODEL_NAME,,}" in
  olmoe)
    MODEL_NAME="olmoe"
    DEFAULT_MODEL="$ROOT/local_models/olmoe_compat"
    ;;
  qwen|qwen2_moe|qwen2-moe)
    MODEL_NAME="qwen2_moe"
    DEFAULT_MODEL="$ROOT/local_models/Qwen1.5-MoE-A2.7B"
    ;;
  *)
    echo "Unsupported MODEL_NAME=$MODEL_NAME. Use olmoe or qwen2_moe." >&2
    exit 2
    ;;
esac

MODEL="${MODEL:-$DEFAULT_MODEL}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/duquant_moe_baseline_${MODEL_NAME}_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/output}"
RUN_LOG="$OUT_ROOT/run.log"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

CALIB_DATASET="${CALIB_DATASET:-wikitext2}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"

W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
ROUTER_W_BITS="${ROUTER_W_BITS:-8}"
ROUTER_A_BITS="${ROUTER_A_BITS:-8}"

# Original DuQuant-style quantization granularity: weight per output channel,
# activation per token. Leave group args unset unless explicitly requested.
W_DYNAMIC_METHOD="${W_DYNAMIC_METHOD:-per_channel}"
ROUTER_W_DYNAMIC_METHOD="${ROUTER_W_DYNAMIC_METHOD:-per_channel}"
A_DYNAMIC_METHOD="${A_DYNAMIC_METHOD:-per_token}"
GROUP_SIZE="${GROUP_SIZE:-}"
ACT_GROUP_SIZE="${ACT_GROUP_SIZE:-}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"

SMOOTH="${SMOOTH:-1}"
ALPHA="${ALPHA:-0.6}"
FC1_SCALE_MERGE="${FC1_SCALE_MERGE:-max}"
ACT_MEAN_BETA="${ACT_MEAN_BETA:-2}"
MOE_DOWN_SMOOTH_MODE="${MOE_DOWN_SMOOTH_MODE:-duquant}"
LAC="${LAC:-0.9}"
SWC="${SWC:-0.8}"

SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-0}"
MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-same}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"
FAST_MOE_DOWN_CALIBRATION="${FAST_MOE_DOWN_CALIBRATION:-0}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
DISABLE_MOE_GATE_UP_DUQUANT_ROTATION="${DISABLE_MOE_GATE_UP_DUQUANT_ROTATION:-0}"

mkdir -p "$OUT_ROOT" "$OUTPUT_DIR" "$CACHE_DIR"

csv_quote() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

extract_last_value() {
  local pattern="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    grep -E "$pattern" "$file" | tail -n 1 | sed -E 's/.*: *([0-9.eE+-]+).*/\1/' || true
  fi
}

extract_task_results_tsv() {
  local file="$1"
  python3 - "$file" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(errors="ignore") if path.exists() else ""
records = []
for line in text.splitlines():
    if "INFO {" not in line or "acc_avg" not in line:
        continue
    payload = line.split("INFO ", 1)[-1]
    try:
        obj = ast.literal_eval(payload)
    except Exception:
        continue
    if isinstance(obj, dict):
        records.append(obj)

result = records[-1] if records else {}

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

task_args=()
if [[ -n "$TASKS" ]]; then
  task_args=(--tasks "$TASKS" --num_fewshot "$NUM_FEWSHOT" --batch_size "$BATCH_SIZE")
fi

smooth_args=()
if [[ "$SMOOTH" == "1" || "$SMOOTH" == "true" || "$SMOOTH" == "TRUE" ]]; then
  smooth_args=(--smooth --fc1_scale_merge "$FC1_SCALE_MERGE" --act_mean_beta "$ACT_MEAN_BETA" --moe_down_smooth_mode "$MOE_DOWN_SMOOTH_MODE" --alpha "$ALPHA")
else
  # main.py validates smooth-dependent score names even when topk=0.
  if [[ "$MOE_OUTLIER_SCORE" == "smooth_scale" || "$MOE_OUTLIER_SCORE" == "layer_shared_scale" || "$MOE_OUTLIER_SCORE" == "shared_scale" ]]; then
    MOE_OUTLIER_SCORE="weight_max"
  fi
fi

clip_args=()
if [[ -n "$LAC" ]]; then
  clip_args+=(--lac "$LAC")
fi
if [[ -n "$SWC" ]]; then
  clip_args+=(--swc "$SWC")
fi

group_args=()
if [[ -n "$GROUP_SIZE" ]]; then
  group_args+=(--group_size "$GROUP_SIZE")
fi
if [[ -n "$ACT_GROUP_SIZE" ]]; then
  group_args+=(--act_group_size "$ACT_GROUP_SIZE")
fi
if [[ -n "$WEIGHT_CHANNEL_GROUP_SIZE" ]]; then
  group_args+=(--weight_channel_group_size "$WEIGHT_CHANNEL_GROUP_SIZE")
fi
if [[ -n "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" ]]; then
  group_args+=(--router_weight_channel_group_size "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE")
fi

gate_up_rotation_args=()
if [[ "$DISABLE_MOE_GATE_UP_DUQUANT_ROTATION" == "1" || "$DISABLE_MOE_GATE_UP_DUQUANT_ROTATION" == "true" || "$DISABLE_MOE_GATE_UP_DUQUANT_ROTATION" == "TRUE" ]]; then
  gate_up_rotation_args=(--disable_moe_gate_up_duquant_rotation)
fi

fast_moe_args=()
if [[ "$FAST_MOE_DOWN_CALIBRATION" == "1" || "$FAST_MOE_DOWN_CALIBRATION" == "true" || "$FAST_MOE_DOWN_CALIBRATION" == "TRUE" ]]; then
  fast_moe_args=(--fast_moe_down_calibration --fast_moe_calib_tokens "$FAST_MOE_CALIB_TOKENS")
fi

printf 'label,model_name,model_path,wbits,abits,router_wbits,router_abits,w_dynamic_method,a_dynamic_method,router_w_dynamic_method,smooth,fc1_scale_merge,moe_down_smooth_mode,lac,swc,moe_outlier_topk,moe_outlier_quant,moe_outlier_score,disable_moe_gate_up_duquant_rotation,fast_moe_down_calibration,status,wikitext2,c4,acc_avg,arc_challenge,arc_easy,boolq,openbookqa,piqa,winogrande,calibration_seconds,smooth_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,output_dir,rank_log,run_log\n' > "$SUMMARY_CSV"

echo "[duquant-moe-baseline] output root: $OUT_ROOT"
echo "[duquant-moe-baseline] model: $MODEL_NAME -> $MODEL"
echo "[duquant-moe-baseline] bits: W${W_BITS}A${A_BITS}, router W${ROUTER_W_BITS}A${ROUTER_A_BITS}"
echo "[duquant-moe-baseline] methods: weight=$W_DYNAMIC_METHOD, act=$A_DYNAMIC_METHOD, router_weight=$ROUTER_W_DYNAMIC_METHOD"
echo "[duquant-moe-baseline] smooth=$SMOOTH fc1_scale_merge=$FC1_SCALE_MERGE moe_down_smooth_mode=$MOE_DOWN_SMOOTH_MODE"
echo "[duquant-moe-baseline] disabled HeteroQuant components: moe_quant_plan, OSE topK unless explicitly overridden, fast MoE calibration unless explicitly enabled"

status="ok"
set +e
python3 main.py \
  --model "$MODEL" \
  --model_name "$MODEL_NAME" \
  --calib_dataset "$CALIB_DATASET" \
  --wbits "$W_BITS" --abits "$A_BITS" \
  --router_wbits "$ROUTER_W_BITS" --router_abits "$ROUTER_A_BITS" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --w_dynamic_method "$W_DYNAMIC_METHOD" \
  --a_dynamic_method "$A_DYNAMIC_METHOD" \
  --router_w_dynamic_method "$ROUTER_W_DYNAMIC_METHOD" \
  --nsamples "$NSAMPLES" \
  --seq_length "$SEQ_LENGTH" \
  --scale_search_steps "$SCALE_SEARCH_STEPS" \
  --eval_ppl \
  --test_dataset "$TEST_DATASET" \
  "${task_args[@]}" \
  "${group_args[@]}" \
  "${smooth_args[@]}" \
  "${clip_args[@]}" \
  --moe_outlier_topk "$MOE_OUTLIER_TOPK" \
  --moe_outlier_quant "$MOE_OUTLIER_QUANT" \
  --moe_outlier_score "$MOE_OUTLIER_SCORE" \
  "${gate_up_rotation_args[@]}" \
  "${fast_moe_args[@]}" \
  2>&1 | tee "$RUN_LOG"
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" -ne 0 ]]; then
  status="run_failed"
fi

rank_log="$(find "$OUTPUT_DIR" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
wikitext2="$(extract_last_value 'wikitext2 :' "$rank_log")"
c4="$(extract_last_value 'c4 :' "$rank_log")"
calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
smooth_seconds="$(extract_last_value '\[timing\] smooth_seconds:' "$rank_log")"
duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"
task_values="$(extract_task_results_tsv "$rank_log")"
IFS=$'\t' read -r acc_avg arc_challenge arc_easy boolq openbookqa piqa winogrande <<< "$task_values"

{
  csv_quote "duquant_moe_baseline"; printf ','
  csv_quote "$MODEL_NAME"; printf ','
  csv_quote "$MODEL"; printf ','
  csv_quote "$W_BITS"; printf ','
  csv_quote "$A_BITS"; printf ','
  csv_quote "$ROUTER_W_BITS"; printf ','
  csv_quote "$ROUTER_A_BITS"; printf ','
  csv_quote "$W_DYNAMIC_METHOD"; printf ','
  csv_quote "$A_DYNAMIC_METHOD"; printf ','
  csv_quote "$ROUTER_W_DYNAMIC_METHOD"; printf ','
  csv_quote "$SMOOTH"; printf ','
  csv_quote "$FC1_SCALE_MERGE"; printf ','
  csv_quote "$MOE_DOWN_SMOOTH_MODE"; printf ','
  csv_quote "$LAC"; printf ','
  csv_quote "$SWC"; printf ','
  csv_quote "$MOE_OUTLIER_TOPK"; printf ','
  csv_quote "$MOE_OUTLIER_QUANT"; printf ','
  csv_quote "$MOE_OUTLIER_SCORE"; printf ','
  csv_quote "$DISABLE_MOE_GATE_UP_DUQUANT_ROTATION"; printf ','
  csv_quote "$FAST_MOE_DOWN_CALIBRATION"; printf ','
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
  csv_quote "$smooth_seconds"; printf ','
  csv_quote "$duquant_seconds"; printf ','
  csv_quote "$inference_wikitext2_seconds"; printf ','
  csv_quote "$inference_c4_seconds"; printf ','
  csv_quote "$OUTPUT_DIR"; printf ','
  csv_quote "$rank_log"; printf ','
  csv_quote "$RUN_LOG"; printf '\n'
} >> "$SUMMARY_CSV"

echo "[duquant-moe-baseline] done status=$status"
echo "[duquant-moe-baseline] summary: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"

if [[ "$status" != "ok" ]]; then
  exit 1
fi
