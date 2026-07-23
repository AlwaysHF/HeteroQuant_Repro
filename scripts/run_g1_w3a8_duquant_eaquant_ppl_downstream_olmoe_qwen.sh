#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/g1_w3a8_duquant_eaquant_ppl_downstream_olmoe_qwen_${RUN_ID}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

OLMOE_MODEL="${OLMOE_MODEL:-$ROOT/local_models/olmoe_compat}"
QWEN_MODEL="${QWEN_MODEL:-$ROOT/local_models/Qwen1.5-MoE-A2.7B}"

CALIB_DATASET="${CALIB_DATASET:-wikitext2}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"

W_BITS="${W_BITS:-3}"
A_BITS="${A_BITS:-8}"
ROUTER_W_BITS="${ROUTER_W_BITS:-8}"
ROUTER_A_BITS="${ROUTER_A_BITS:-8}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-1}"
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

append_summary() {
  local label="$1" method="$2" model_name="$3" model_path="$4" status="$5"
  local output_dir="$6" run_log="$7" rank_log="$8"
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
    csv_quote "$method"; printf ','
    csv_quote "$model_name"; printf ','
    csv_quote "$model_path"; printf ','
    csv_quote "$W_BITS"; printf ','
    csv_quote "$A_BITS"; printf ','
    csv_quote "$ROUTER_W_BITS"; printf ','
    csv_quote "$ROUTER_A_BITS"; printf ','
    csv_quote "$WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$TEST_DATASET"; printf ','
    csv_quote "$TASKS"; printf ','
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
    csv_quote "$inference_task_seconds"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$rank_log"; printf ','
    csv_quote "$run_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

printf 'label,method,model_name,model_path,wbits,abits,router_wbits,router_abits,weight_channel_group_size,router_weight_channel_group_size,test_dataset,tasks,status,wikitext2,c4,acc_avg,arc_challenge,arc_easy,boolq,openbookqa,piqa,winogrande,calibration_seconds,smooth_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,inference_task_seconds,output_dir,rank_log,run_log\n' > "$SUMMARY_CSV"

run_one() {
  local method="$1"
  local model_name="$2"
  local model_path="$3"
  local label="${model_name}_${method}_g${WEIGHT_CHANNEL_GROUP_SIZE}_w${W_BITS}a${A_BITS}_ppl_downstream"
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
  echo "[g1-w3a8] start $label"
  echo "[g1-w3a8] model=$model_path"
  echo "[g1-w3a8] eval: ppl=$TEST_DATASET tasks=$TASKS"
  echo "[g1-w3a8] quant: W${W_BITS}A${A_BITS}, router W${ROUTER_W_BITS}A${ROUTER_A_BITS}, group=$WEIGHT_CHANNEL_GROUP_SIZE"
  echo "[g1-w3a8] method=$method w_dynamic=$w_dynamic_method a_dynamic=$a_dynamic_method router_w_dynamic=$router_w_dynamic_method"

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
      --eval_ppl \
      --test_dataset "$TEST_DATASET" \
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
  echo "[g1-w3a8] finished $label status=$status"
  echo "[g1-w3a8] summary: $SUMMARY_CSV"
}

echo "[g1-w3a8] output root: $OUT_ROOT"
echo "[g1-w3a8] four runs: olmoe/qwen2_moe x duquant/eaquant"
echo "[g1-w3a8] PPL datasets: $TEST_DATASET"
echo "[g1-w3a8] downstream tasks: $TASKS"

run_one "duquant" "olmoe" "$OLMOE_MODEL"
run_one "eaquant" "olmoe" "$OLMOE_MODEL"
run_one "duquant" "qwen2_moe" "$QWEN_MODEL"
run_one "eaquant" "qwen2_moe" "$QWEN_MODEL"

echo
echo "[g1-w3a8] all done"
echo "[g1-w3a8] summary: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
