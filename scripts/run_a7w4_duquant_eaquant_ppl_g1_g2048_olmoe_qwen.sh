#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/a7w4_duquant_eaquant_ppl_g1_g2048_olmoe_qwen_${RUN_ID}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

OLMOE_MODEL="${OLMOE_MODEL:-$ROOT/local_models/olmoe_compat}"
QWEN_MODEL="${QWEN_MODEL:-$ROOT/local_models/Qwen1.5-MoE-A2.7B}"

CALIB_DATASET="${CALIB_DATASET:-wikitext2}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"

W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-7}"
ROUTER_W_BITS="${ROUTER_W_BITS:-8}"
ROUTER_A_BITS="${ROUTER_A_BITS:-8}"
GROUP_SIZES="${GROUP_SIZES:-1,2048}"

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

extract_ppl_tsv() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import ast
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(errors="ignore") if path.exists() else ""
result = {}
for line in text.splitlines():
    if "INFO {" in line and ("wikitext2" in line or "c4" in line):
        payload = line.split("INFO ", 1)[-1]
        try:
            obj = ast.literal_eval(payload)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for key in ("wikitext2", "c4"):
                if key in obj:
                    result[key] = obj[key]
    for key in ("wikitext2", "c4"):
        match = re.search(rf"{re.escape(key)}\s*:\s*([0-9.eE+-]+)", line)
        if match:
            result[key] = match.group(1)

print("\t".join(str(result.get(key, "")) for key in ("wikitext2", "c4")))
PY
}

append_summary() {
  local label="$1" method="$2" model_name="$3" model_path="$4" group_size="$5"
  local status="$6" output_dir="$7" run_log="$8" rank_log="$9"
  local values wikitext2 c4 calibration_seconds smooth_seconds duquant_seconds inference_wikitext2_seconds inference_c4_seconds

  values="$(extract_ppl_tsv "$rank_log")"
  IFS=$'\t' read -r wikitext2 c4 <<< "$values"
  calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
  smooth_seconds="$(extract_last_value '\[timing\] smooth_seconds:' "$rank_log")"
  duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
  inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
  inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"

  {
    csv_quote "$label"; printf ','
    csv_quote "$method"; printf ','
    csv_quote "$model_name"; printf ','
    csv_quote "$model_path"; printf ','
    csv_quote "$W_BITS"; printf ','
    csv_quote "$A_BITS"; printf ','
    csv_quote "$((W_BITS + A_BITS))"; printf ','
    csv_quote "$ROUTER_W_BITS"; printf ','
    csv_quote "$ROUTER_A_BITS"; printf ','
    csv_quote "$group_size"; printf ','
    csv_quote "$group_size"; printf ','
    csv_quote "$TEST_DATASET"; printf ','
    csv_quote "$status"; printf ','
    csv_quote "$wikitext2"; printf ','
    csv_quote "$c4"; printf ','
    csv_quote "$calibration_seconds"; printf ','
    csv_quote "$smooth_seconds"; printf ','
    csv_quote "$duquant_seconds"; printf ','
    csv_quote "$inference_wikitext2_seconds"; printf ','
    csv_quote "$inference_c4_seconds"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$rank_log"; printf ','
    csv_quote "$run_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

printf 'label,method,model_name,model_path,wbits,abits,total_wa_bits,router_wbits,router_abits,weight_channel_group_size,router_weight_channel_group_size,test_dataset,status,wikitext2,c4,calibration_seconds,smooth_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,output_dir,rank_log,run_log\n' > "$SUMMARY_CSV"

run_one() {
  local method="$1"
  local model_name="$2"
  local model_path="$3"
  local group_size="$4"
  local label="${model_name}_${method}_g${group_size}_a${A_BITS}w${W_BITS}_ppl"
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
  local gate_up_rotation_args=()

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
      gate_up_rotation_args=(--disable_moe_gate_up_duquant_rotation)
      ;;
    *)
      echo "unknown method: $method" >&2
      exit 2
      ;;
  esac

  echo
  echo "[a7w4-ppl] start $label"
  echo "[a7w4-ppl] model=$model_path"
  echo "[a7w4-ppl] eval: ppl=$TEST_DATASET"
  echo "[a7w4-ppl] quant: A${A_BITS}W${W_BITS}, router A${ROUTER_A_BITS}W${ROUTER_W_BITS}, group=$group_size"
  echo "[a7w4-ppl] method=$method w_dynamic=$w_dynamic_method a_dynamic=$a_dynamic_method router_w_dynamic=$router_w_dynamic_method"

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
      --batch_size "$BATCH_SIZE" \
      --group_size -1 \
      --weight_channel_group_size "$group_size" \
      --router_weight_channel_group_size "$group_size" \
      --act_group_size -1 \
      --smooth \
      --fc1_scale_merge "$fc1_scale_merge" \
      --moe_down_smooth_mode "$moe_down_smooth_mode" \
      --alpha "$alpha" \
      --lac "$lac" \
      --swc "$swc" \
      --moe_outlier_topk 0 \
      --moe_outlier_quant same \
      --moe_outlier_score smooth_scale \
      "${gate_up_rotation_args[@]}"; then
    status="run_failed"
  fi

  local rank_log
  rank_log="$(find "$output_dir" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
  append_summary "$label" "$method" "$model_name" "$model_path" "$group_size" "$status" "$output_dir" "$run_log" "$rank_log"
  echo "[a7w4-ppl] finished $label status=$status"
  echo "[a7w4-ppl] summary: $SUMMARY_CSV"
}

echo "[a7w4-ppl] output root: $OUT_ROOT"
echo "[a7w4-ppl] eight runs: olmoe/qwen2_moe x duquant/eaquant x group sizes $GROUP_SIZES"
echo "[a7w4-ppl] PPL datasets: $TEST_DATASET"
echo "[a7w4-ppl] quant: A${A_BITS}W${W_BITS}; router A${ROUTER_A_BITS}W${ROUTER_W_BITS}; no OSE; no tier allocation"

IFS=',' read -ra GROUP_ARRAY <<< "$GROUP_SIZES"
for group_size in "${GROUP_ARRAY[@]}"; do
  run_one "duquant" "olmoe" "$OLMOE_MODEL" "$group_size"
  run_one "eaquant" "olmoe" "$OLMOE_MODEL" "$group_size"
  run_one "duquant" "qwen2_moe" "$QWEN_MODEL" "$group_size"
  run_one "eaquant" "qwen2_moe" "$QWEN_MODEL" "$group_size"
done

echo
echo "[a7w4-ppl] all done"
echo "[a7w4-ppl] summary: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
