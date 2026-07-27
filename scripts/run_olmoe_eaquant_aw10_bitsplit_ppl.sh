#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/olmoe_eaquant_aw10_bitsplit_ppl_${RUN_ID}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

MODEL="${MODEL:-$ROOT/local_models/olmoe_compat}"
CALIB_DATASET="${CALIB_DATASET:-wikitext2}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-6}"

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

append_summary() {
  local label="$1" wbits="$2" abits="$3" status="$4" output_dir="$5" run_log="$6" rank_log="$7"
  local wikitext2 c4 calibration_seconds smooth_seconds duquant_seconds inference_wikitext2_seconds inference_c4_seconds

  wikitext2="$(extract_last_value 'wikitext2 :' "$rank_log")"
  c4="$(extract_last_value 'c4 :' "$rank_log")"
  calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
  smooth_seconds="$(extract_last_value '\[timing\] smooth_seconds:' "$rank_log")"
  duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
  inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
  inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"

  {
    csv_quote "$label"; printf ','
    csv_quote "eaquant"; printf ','
    csv_quote "olmoe"; printf ','
    csv_quote "$MODEL"; printf ','
    csv_quote "$wbits"; printf ','
    csv_quote "$abits"; printf ','
    csv_quote "$((wbits + abits))"; printf ','
    csv_quote "$ROUTER_W_BITS"; printf ','
    csv_quote "$ROUTER_A_BITS"; printf ','
    csv_quote "$WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
    csv_quote "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE"; printf ','
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
  local label="$1" wbits="$2" abits="$3"
  local case_dir="$OUT_ROOT/$label"
  local output_dir="$case_dir/output"
  local run_log="$case_dir/run.log"

  mkdir -p "$output_dir"

  echo
  echo "[olmoe-eaquant-aw10] start $label: W${wbits}A${abits}"
  echo "[olmoe-eaquant-aw10] model=$MODEL"
  echo "[olmoe-eaquant-aw10] PPL=$TEST_DATASET group=$WEIGHT_CHANNEL_GROUP_SIZE"

  local status="ok"
  if ! run_logged "$run_log" \
    python3 main.py \
      --model "$MODEL" \
      --model_name olmoe \
      --calib_dataset "$CALIB_DATASET" \
      --wbits "$wbits" --abits "$abits" \
      --router_wbits "$ROUTER_W_BITS" --router_abits "$ROUTER_A_BITS" \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$output_dir" \
      --w_dynamic_method per_channel_tensor \
      --a_dynamic_method per_tensor \
      --router_w_dynamic_method per_channel_kl_top0 \
      --nsamples "$NSAMPLES" \
      --seq_length "$SEQ_LENGTH" \
      --scale_search_steps "$SCALE_SEARCH_STEPS" \
      --eval_ppl \
      --test_dataset "$TEST_DATASET" \
      --batch_size "$BATCH_SIZE" \
      --group_size -1 \
      --weight_channel_group_size "$WEIGHT_CHANNEL_GROUP_SIZE" \
      --router_weight_channel_group_size "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
      --act_group_size -1 \
      --smooth \
      --fc1_scale_merge max \
      --moe_down_smooth_mode otsu \
      --alpha 0.6 \
      --lac 0.9 \
      --swc 0.8 \
      --moe_outlier_topk 0 \
      --moe_outlier_quant same \
      --moe_outlier_score smooth_scale; then
    status="run_failed"
  fi

  local rank_log
  rank_log="$(find "$output_dir" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
  append_summary "$label" "$wbits" "$abits" "$status" "$output_dir" "$run_log" "$rank_log"
  echo "[olmoe-eaquant-aw10] finished $label status=$status"
  echo "[olmoe-eaquant-aw10] summary: $SUMMARY_CSV"
}

echo "[olmoe-eaquant-aw10] output root: $OUT_ROOT"
echo "[olmoe-eaquant-aw10] cases: A8W2, A7W3, A6W4"
echo "[olmoe-eaquant-aw10] EAQuant config: a_dynamic=per_tensor, w_dynamic=per_channel_tensor, fc1=max, OSE/tier off"

run_one "a8w2" 2 8
run_one "a7w3" 3 7
run_one "a6w4" 4 6

echo
echo "[olmoe-eaquant-aw10] all done"
echo "[olmoe-eaquant-aw10] summary: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
