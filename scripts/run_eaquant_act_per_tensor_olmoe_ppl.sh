#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/eaquant_act_per_tensor_olmoe_ppl_${RUN_ID}}"
OUTPUT_DIR="$OUT_ROOT/output"
RUN_LOG="$OUT_ROOT/run.log"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

MODEL="${MODEL:-$ROOT/local_models/olmoe_compat}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"

# Keep the current EAQuant-like comparison at W4A8 by default. Set A_BITS=4
# when you want to match EAQuant run.sh's W4A4 setting.
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-8}"
ROUTER_W_BITS="${ROUTER_W_BITS:-8}"
ROUTER_A_BITS="${ROUTER_A_BITS:-8}"

mkdir -p "$OUT_ROOT" "$OUTPUT_DIR"

extract_last_value() {
  local pattern="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    grep -E "$pattern" "$file" | tail -n 1 | sed -E 's/.*: *([0-9.eE+-]+).*/\1/' || true
  fi
}

csv_quote() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

append_summary() {
  local status="$1" rank_log="$2"
  local wikitext2 c4 calibration_seconds duquant_seconds inference_wikitext2_seconds inference_c4_seconds
  wikitext2="$(extract_last_value 'wikitext2 :' "$rank_log")"
  c4="$(extract_last_value 'c4 :' "$rank_log")"
  calibration_seconds="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
  duquant_seconds="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
  inference_wikitext2_seconds="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
  inference_c4_seconds="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"

  {
    csv_quote "eaquant_act_per_tensor"; printf ','
    csv_quote "per_tensor"; printf ','
    csv_quote "$W_BITS"; printf ','
    csv_quote "$A_BITS"; printf ','
    csv_quote "$ROUTER_W_BITS"; printf ','
    csv_quote "$ROUTER_A_BITS"; printf ','
    csv_quote "max"; printf ','
    csv_quote "0"; printf ','
    csv_quote "0"; printf ','
    csv_quote "0"; printf ','
    csv_quote "1"; printf ','
    csv_quote "1"; printf ','
    csv_quote "$status"; printf ','
    csv_quote "$wikitext2"; printf ','
    csv_quote "$c4"; printf ','
    csv_quote "$calibration_seconds"; printf ','
    csv_quote "$duquant_seconds"; printf ','
    csv_quote "$inference_wikitext2_seconds"; printf ','
    csv_quote "$inference_c4_seconds"; printf ','
    csv_quote "$OUTPUT_DIR"; printf ','
    csv_quote "$rank_log"; printf ','
    csv_quote "$RUN_LOG"; printf '\n'
  } >> "$SUMMARY_CSV"
}

printf 'label,a_dynamic_method,wbits,abits,router_wbits,router_abits,fc1_scale_merge,fast_moe_down_calibration,moe_outlier_topk,disable_moe_gate_up_duquant_rotation,weight_channel_group_size,router_weight_channel_group_size,status,wikitext2,c4,calibration_seconds,duquant_seconds,inference_wikitext2_seconds,inference_c4_seconds,output_dir,rank_log,run_log\n' > "$SUMMARY_CSV"

echo "[eaquant-act-per-tensor] output root: $OUT_ROOT"
echo "[eaquant-act-per-tensor] model: $MODEL"
echo "[eaquant-act-per-tensor] test_dataset: $TEST_DATASET"
echo "[eaquant-act-per-tensor] tasks: ${TASKS:-<none>}"
echo "[eaquant-act-per-tensor] activation dynamic method: per_tensor"
echo "[eaquant-act-per-tensor] EAQuant-like: plan off, OSE off, MoE gate/up DuQuant rotation on, fast MoE calibration off, fc1 max smooth"

status="ok"
if ! (
  PLAN_PATH="" \
  MOE_OUTLIER_TOPK=0 \
  DISABLE_MOE_GATE_UP_DUQUANT_ROTATION=0 \
  FAST_MOE_DOWN_CALIBRATION=0 \
  FC1_SCALE_MERGE=max \
  MOE_DOWN_SMOOTH_MODE=otsu \
  LAC=0.9 \
  SWC=0.8 \
  A_DYNAMIC_METHOD=per_tensor \
  W_BITS="$W_BITS" \
  A_BITS="$A_BITS" \
  ROUTER_W_BITS="$ROUTER_W_BITS" \
  ROUTER_A_BITS="$ROUTER_A_BITS" \
  WEIGHT_CHANNEL_GROUP_SIZE=1 \
  ROUTER_WEIGHT_CHANNEL_GROUP_SIZE=1 \
  MODEL="$MODEL" \
  CACHE_DIR="$CACHE_DIR" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  TEST_DATASET="$TEST_DATASET" \
  TASKS="$TASKS" \
  NSAMPLES="$NSAMPLES" \
  SEQ_LENGTH="$SEQ_LENGTH" \
  SCALE_SEARCH_STEPS="$SCALE_SEARCH_STEPS" \
  bash scripts/reproduce_olmoe_691.sh
) 2>&1 | tee "$RUN_LOG"; then
  status="run_failed"
fi

rank_log="$(find "$OUTPUT_DIR" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
append_summary "$status" "$rank_log"

echo "[eaquant-act-per-tensor] done status=$status"
echo "[eaquant-act-per-tensor] summary: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"

if [[ "$status" != "ok" ]]; then
  exit 1
fi
