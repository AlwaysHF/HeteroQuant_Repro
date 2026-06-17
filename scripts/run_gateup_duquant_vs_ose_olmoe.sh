#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/experiments/gateup_duquant_vs_ose_olmoe_${TIMESTAMP}}"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
MASTER_LOG="$RUN_ROOT/master.log"

TEST_DATASET="${TEST_DATASET:-wikitext2}"
TASKS="${TASKS-}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
ETA_INTERVAL_SECONDS="${ETA_INTERVAL_SECONDS:-60}"
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-900}"
LAYERS_TOTAL="${LAYERS_TOTAL:-16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$RUN_ROOT"

format_duration() {
  local seconds="$1"
  if (( seconds < 0 )); then seconds=0; fi
  local h=$((seconds / 3600))
  local m=$(((seconds % 3600) / 60))
  local s=$((seconds % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

log_master() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "$msg" | tee -a "$MASTER_LOG"
}

extract_metric() {
  local log_file="$1"
  local kind="$2"
  local line=""
  case "$kind" in
    ppl_wikitext2)
      line="$(grep -E 'INFO wikitext2[[:space:]]*:' "$log_file" | tail -n 1 || true)"
      sed -nE 's/.*wikitext2[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*/\1/p' <<<"$line"
      ;;
    duquant_seconds)
      line="$(grep -E '\[timing\] duquant_seconds:' "$log_file" | tail -n 1 || true)"
      sed -nE 's/.*duquant_seconds:[[:space:]]*([0-9.eE+-]+).*/\1/p' <<<"$line"
      ;;
    calibration_seconds)
      line="$(grep -E '\[timing\] calibration_seconds:' "$log_file" | tail -n 1 || true)"
      sed -nE 's/.*calibration_seconds:[[:space:]]*([0-9.eE+-]+).*/\1/p' <<<"$line"
      ;;
    inference_seconds_wikitext2)
      line="$(grep -E '\[timing\] inference_seconds\.wikitext2:' "$log_file" | tail -n 1 || true)"
      sed -nE 's/.*inference_seconds\.wikitext2:[[:space:]]*([0-9.eE+-]+).*/\1/p' <<<"$line"
      ;;
  esac
}

last_layer_seen() {
  local log_file="$1"
  local layer=""
  layer="$(grep -aoE 'Start quantize layer [0-9]+' "$log_file" | tail -n 1 | awk '{print $4}' || true)"
  if [[ -n "$layer" ]]; then
    echo "$layer"
  fi
}

run_variant() {
  local index="$1"
  local total="$2"
  local label="$3"
  local disable_gateup="$4"
  local topk="$5"
  local quant="$6"
  local shared_score="$7"
  local completed_runs="$8"
  local completed_seconds="$9"

  local estimate="$EST_SECONDS_PER_RUN"
  if (( completed_runs > 0 )); then
    estimate=$((completed_seconds / completed_runs))
    if (( estimate <= 0 )); then estimate="$EST_SECONDS_PER_RUN"; fi
  fi

  local variant_dir="$RUN_ROOT/$label"
  local output_dir="$variant_dir/heteroquant_a8w4"
  local run_log="$variant_dir/run.log"
  mkdir -p "$variant_dir"

  local variants_left_after=$((total - index))
  local estimated_total_remaining=$((estimate * (variants_left_after + 1)))
  log_master "variant $index/$total start: $label disable_gateup=$disable_gateup topk=$topk quant=$quant shared_score=$shared_score"
  log_master "estimated remaining before this variant: $(format_duration "$estimated_total_remaining")"
  log_master "variant log: $run_log"

  local start_ts
  start_ts="$(date +%s)"

  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    TASKS="$TASKS" \
    TEST_DATASET="$TEST_DATASET" \
    NSAMPLES="$NSAMPLES" \
    SEQ_LENGTH="$SEQ_LENGTH" \
    FAST_MOE_CALIB_TOKENS="$FAST_MOE_CALIB_TOKENS" \
    DISABLE_MOE_GATE_UP_DUQUANT_ROTATION="$disable_gateup" \
    MOE_OUTLIER_TOPK="$topk" \
    MOE_OUTLIER_QUANT="$quant" \
    MOE_OUTLIER_SCORE="$shared_score" \
    OUTPUT_DIR="$output_dir" \
    bash scripts/reproduce_olmoe_691.sh
  ) >"$run_log" 2>&1 &
  local run_pid=$!

  tail -n 0 -F "$run_log" &
  local tail_pid=$!

  while kill -0 "$run_pid" 2>/dev/null; do
    sleep "$ETA_INTERVAL_SECONDS"
    if ! kill -0 "$run_pid" 2>/dev/null; then
      break
    fi
    local now elapsed current_eta total_eta layer layer_text
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    current_eta=$((estimate - elapsed))
    if (( current_eta < 0 )); then current_eta=0; fi
    total_eta=$((current_eta + variants_left_after * estimate))
    layer="$(last_layer_seen "$run_log")"
    layer_text=""
    if [[ -n "$layer" ]]; then
      layer_text=" layer=$((layer + 1))/$LAYERS_TOTAL"
    fi
    log_master "ETA $label: elapsed=$(format_duration "$elapsed") current~$(format_duration "$current_eta") total~$(format_duration "$total_eta")$layer_text"
  done

  local exit_code=0
  set +e
  wait "$run_pid"
  exit_code=$?
  set -e
  kill "$tail_pid" >/dev/null 2>&1 || true
  wait "$tail_pid" >/dev/null 2>&1 || true

  local end_ts elapsed ppl duq calib infer
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  ppl="$(extract_metric "$run_log" ppl_wikitext2)"
  duq="$(extract_metric "$run_log" duquant_seconds)"
  calib="$(extract_metric "$run_log" calibration_seconds)"
  infer="$(extract_metric "$run_log" inference_seconds_wikitext2)"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$label" "$disable_gateup" "$topk" "$quant" "$shared_score" "$exit_code" "$elapsed" \
    "$ppl" "$duq" "$calib" "$infer" "$output_dir" "$run_log" >>"$SUMMARY_CSV"

  log_master "variant $label done: exit=$exit_code elapsed=$(format_duration "$elapsed") ppl_wikitext2=${ppl:-NA} inference_seconds=${infer:-NA}"
  LAST_ELAPSED="$elapsed"
  if (( exit_code != 0 )); then
    log_master "stopping because variant $label failed"
    exit "$exit_code"
  fi
}

main() {
  log_master "gate/up DuQuant vs OSE ablation root: $RUN_ROOT"
  log_master "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES TEST_DATASET=$TEST_DATASET TASKS='${TASKS}' NSAMPLES=$NSAMPLES SEQ_LENGTH=$SEQ_LENGTH"
  log_master "ETA interval: ${ETA_INTERVAL_SECONDS}s, initial per-run estimate: $(format_duration "$EST_SECONDS_PER_RUN")"

  printf 'variant,disable_gateup_duquant,topk,quant,shared_score,exit_code,elapsed_seconds,ppl_wikitext2,duquant_seconds,calibration_seconds,inference_seconds_wikitext2,output_dir,log\n' >"$SUMMARY_CSV"

  local variants=(
    "full_duquant|0|0|same|none"
    "no_gateup_no_ose|1|0|same|none"
    "no_gateup_ose_fp16|1|64|fp16|smooth_scale"
    "no_gateup_ose_w8a8|1|64|w8a8|smooth_scale"
  )
  if [[ "${INCLUDE_EXTRA:-0}" == "1" ]]; then
    variants+=(
      "full_duquant_ose_w8a8|0|64|w8a8|smooth_scale"
      "no_gateup_ose_w16a16|1|64|w16a16|smooth_scale"
    )
  fi

  local total="${#variants[@]}"
  local completed_runs=0
  local completed_seconds=0
  local i=1
  for spec in "${variants[@]}"; do
    IFS='|' read -r label disable_gateup topk quant shared_score <<<"$spec"
    run_variant "$i" "$total" "$label" "$disable_gateup" "$topk" "$quant" "$shared_score" "$completed_runs" "$completed_seconds"
    local elapsed="$LAST_ELAPSED"
    completed_runs=$((completed_runs + 1))
    completed_seconds=$((completed_seconds + elapsed))
    i=$((i + 1))
  done

  log_master "all variants done"
  log_master "summary: $SUMMARY_CSV"
  column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
}

main "$@"
