#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/g2048_w4a8_duquant_eaquant_ob_olmoe_qwen_${RUN_ID}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

OLMOE_MODEL="${OLMOE_MODEL:-$ROOT/local_models/olmoe_compat}"
QWEN_MODEL="${QWEN_MODEL:-$ROOT/local_models/Qwen1.5-MoE-A2.7B}"

CALIB_DATASET="${CALIB_DATASET:-wikitext2}"
TASKS="${TASKS:-openbookqa}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"

W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-8}"
ROUTER_W_BITS="${ROUTER_W_BITS:-8}"
ROUTER_A_BITS="${ROUTER_A_BITS:-8}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-2048}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"

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
    task_value("openbookqa"),
]
print("\t".join("" if value is None else str(value) for value in fields))
PY
}

append_summary() {
  local label="$1" method="$2" model_name="$3" model_path="$4" status="$5"
  local output_dir="$6" run_log="$7" rank_log="$8"
  local acc_avg openbookqa calibration_seconds smooth_seconds duquant_seconds inference_task_seconds
  local values
  values="$(extract_task_results_tsv "$rank_log")"
  IFS=$'\t' read -r acc_avg openbookqa <<< "$values"
  calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
  smooth_seconds="$(extract_last_value '\[timing\] smooth_seconds:' "$rank_log")"
  duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
  inference_task_seconds="$(extract_last_value '\[timing\] inference_seconds\.task\.all:' "$rank_log")"

  {
    csv_quote "$label"; printf ','
    csv_quote "$method"; printf ','
    csv_quote "$model_name"; printf ','
    csv_quote "$model_path"; printf ','
    csv_quote "$W_BITS"; printf ','
    csv_quote "$A_BITS"; printf ','
    csv_quote "$ROUTER_W_BITS"; printf ','
    csv_quote "$ROUTER_A_BITS"; printf ','
    csv_quote "$WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$TASKS"; printf ','
    csv_quote "$status"; printf ','
    csv_quote "$acc_avg"; printf ','
    csv_quote "$openbookqa"; printf ','
    csv_quote "$calibration_seconds"; printf ','
    csv_quote "$smooth_seconds"; printf ','
    csv_quote "$duquant_seconds"; printf ','
    csv_quote "$inference_task_seconds"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$rank_log"; printf ','
    csv_quote "$run_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

printf 'label,method,model_name,model_path,wbits,abits,router_wbits,router_abits,weight_channel_group_size,router_weight_channel_group_size,tasks,status,acc_avg,openbookqa,calibration_seconds,smooth_seconds,duquant_seconds,inference_task_seconds,output_dir,rank_log,run_log\n' > "$SUMMARY_CSV"

run_one() {
  local method="$1"
  local model_name="$2"
  local model_path="$3"
  local label="${model_name}_${method}_g${WEIGHT_CHANNEL_GROUP_SIZE}_w${W_BITS}a${A_BITS}_ob"
  local case_dir="$OUT_ROOT/$label"
  local output_dir="$case_dir/output"
  local run_log="$case_dir/run.log"

  mkdir -p "$output_dir"

  local w_dynamic_method
  local a_dynamic_method
  local router_w_dynamic_method
  local fc1_scale_merge
  local moe_down_smooth_mode
  local lac
  local swc
  local alpha

  case "$method" in
    duquant)
      w_dynamic_method="per_channel"
      a_dynamic_method="per_token"
      router_w_dynamic_method="per_channel"
      fc1_scale_merge="max"
      moe_down_smooth_mode="duquant"
      lac="0.9"
      swc="0.8"
      alpha="0.6"
      ;;
    eaquant)
      w_dynamic_method="per_channel_tensor"
      a_dynamic_method="per_tensor"
      router_w_dynamic_method="per_channel_kl_top0"
      fc1_scale_merge="max"
      moe_down_smooth_mode="otsu"
      lac="0.9"
      swc="0.8"
      alpha="0.6"
      ;;
    *)
      echo "unknown method: $method" >&2
      exit 2
      ;;
  esac

  echo
  echo "[g2048-ob] start $label"
  echo "[g2048-ob] model=$model_path"
  echo "[g2048-ob] quant: W${W_BITS}A${A_BITS}, router W${ROUTER_W_BITS}A${ROUTER_A_BITS}, group=$WEIGHT_CHANNEL_GROUP_SIZE"
  echo "[g2048-ob] method=$method w_dynamic=$w_dynamic_method a_dynamic=$a_dynamic_method router_w_dynamic=$router_w_dynamic_method"

  local status="ok"
  if ! run_logged "$run_log" \
    python3 main.py \
      --model "$model_path" \
      --model_name "$model_name" \
      --calib_dataset "$CALIB_DATASET" \
      --wbits "$W_BITS" --abits "$A_BITS" \
      --router_wbits "$ROUTER_W_BITS" --router_abits "$ROUTER_A_BITS" \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$output_dir" \
      --w_dynamic_method "$w_dynamic_method" \
      --a_dynamic_method "$a_dynamic_method" \
      --router_w_dynamic_method "$router_w_dynamic_method" \
      --nsamples "$NSAMPLES" \
      --seq_length "$SEQ_LENGTH" \
      --scale_search_steps "$SCALE_SEARCH_STEPS" \
      --tasks "$TASKS" \
      --num_fewshot "$NUM_FEWSHOT" \
      --batch_size "$BATCH_SIZE" \
      --weight_channel_group_size "$WEIGHT_CHANNEL_GROUP_SIZE" \
      --router_weight_channel_group_size "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
      --smooth \
      --fc1_scale_merge "$fc1_scale_merge" \
      --moe_down_smooth_mode "$moe_down_smooth_mode" \
      --alpha "$alpha" \
      --lac "$lac" \
      --swc "$swc" \
      --moe_outlier_topk 0 \
      --moe_outlier_quant same \
      --moe_outlier_score smooth_scale; then
    status="run_failed"
  fi

  local rank_log
  rank_log="$(find "$output_dir" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
  append_summary "$label" "$method" "$model_name" "$model_path" "$status" "$output_dir" "$run_log" "$rank_log"
  echo "[g2048-ob] finished $label status=$status"
  echo "[g2048-ob] summary: $SUMMARY_CSV"
}

echo "[g2048-ob] output root: $OUT_ROOT"
echo "[g2048-ob] tasks: $TASKS"
echo "[g2048-ob] four runs: olmoe/qwen2_moe x duquant/eaquant"

run_one "duquant" "olmoe" "$OLMOE_MODEL"
run_one "eaquant" "olmoe" "$OLMOE_MODEL"
run_one "duquant" "qwen2_moe" "$QWEN_MODEL"
run_one "eaquant" "qwen2_moe" "$QWEN_MODEL"

echo
echo "[g2048-ob] all done"
echo "[g2048-ob] summary: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
