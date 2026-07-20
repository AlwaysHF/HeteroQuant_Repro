#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/gateup_duquant_vs_ose_repeated_olmoe_${RUN_ID}}"
MODEL="${MODEL:-$ROOT/local_models/olmoe_compat}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"

default_plan="$(find "$ROOT/experiments" -path '*/full_method/plans/moe_quant_plan_routed_tail.json' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
if [[ -z "$default_plan" ]]; then
  default_plan="$(find "$ROOT/experiments" -path '*/3tier_w4a4_w5a8_w6a8/plans/moe_quant_plan_routed_tail.json' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
fi
if [[ -z "$default_plan" ]]; then
  default_plan="$ROOT/plans/olmoe_routed_range/moe_quant_plan_routed_range.json"
fi
PLAN_PATH="${PLAN_PATH:-$default_plan}"

TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
REPEATS="${REPEATS:-3}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SCALE_SEARCH_STEPS="${SCALE_SEARCH_STEPS:-100}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
FAST_MOE_DOWN_CALIBRATION="${FAST_MOE_DOWN_CALIBRATION:-1}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-2048}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"
VARIANTS="${VARIANTS:-full_duquant,no_gateup_no_ose,ours_ose_w8a8,ours_ose_fp16}"

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

printf 'variant,repeat,disable_gateup_duquant,moe_outlier_topk,moe_outlier_quant,moe_outlier_score,status,wikitext2,c4,smooth_seconds,duquant_seconds,calibration_seconds,inference_wikitext2_seconds,inference_c4_seconds,elapsed_seconds,plan_path,output_dir,run_log,rank_log\n' > "$SUMMARY_CSV"

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

variant_spec() {
  case "$1" in
    full_duquant)
      echo "0|0|same|smooth_scale"
      ;;
    no_gateup_no_ose)
      echo "1|0|same|smooth_scale"
      ;;
    ours_ose_w8a8)
      echo "1|64|w8a8|smooth_scale"
      ;;
    ours_ose_fp16)
      echo "1|64|fp16|smooth_scale"
      ;;
    full_duquant_ose_w8a8)
      echo "0|64|w8a8|smooth_scale"
      ;;
    *)
      echo "unknown variant: $1" >&2
      return 1
      ;;
  esac
}

append_summary() {
  local variant="$1" repeat="$2" disable_gateup="$3" topk="$4" quant="$5" score="$6" status="$7"
  local wikitext2="$8" c4="$9" smooth="${10}" duquant="${11}" calibration="${12}" infer_wiki="${13}" infer_c4="${14}" elapsed="${15}" output_dir="${16}" run_log="${17}" rank_log="${18}"
  {
    csv_quote "$variant"; printf ','
    csv_quote "$repeat"; printf ','
    csv_quote "$disable_gateup"; printf ','
    csv_quote "$topk"; printf ','
    csv_quote "$quant"; printf ','
    csv_quote "$score"; printf ','
    csv_quote "$status"; printf ','
    csv_quote "$wikitext2"; printf ','
    csv_quote "$c4"; printf ','
    csv_quote "$smooth"; printf ','
    csv_quote "$duquant"; printf ','
    csv_quote "$calibration"; printf ','
    csv_quote "$infer_wiki"; printf ','
    csv_quote "$infer_c4"; printf ','
    csv_quote "$elapsed"; printf ','
    csv_quote "$PLAN_PATH"; printf ','
    csv_quote "$output_dir"; printf ','
    csv_quote "$run_log"; printf ','
    csv_quote "$rank_log"; printf '\n'
  } >> "$SUMMARY_CSV"
}

echo "[gateup-duquant-vs-ose-repeat] output: $OUT_ROOT"
echo "[gateup-duquant-vs-ose-repeat] model: $MODEL"
echo "[gateup-duquant-vs-ose-repeat] plan: $PLAN_PATH"
echo "[gateup-duquant-vs-ose-repeat] datasets: $TEST_DATASET"
echo "[gateup-duquant-vs-ose-repeat] tasks: ${TASKS:-<none>}"
echo "[gateup-duquant-vs-ose-repeat] variants: $VARIANTS"
echo "[gateup-duquant-vs-ose-repeat] repeats: $REPEATS"

IFS=',' read -ra variant_array <<< "$VARIANTS"
for repeat in $(seq 1 "$REPEATS"); do
  for variant in "${variant_array[@]}"; do
    variant="${variant//[[:space:]]/}"
    [[ -n "$variant" ]] || continue
    IFS='|' read -r disable_gateup topk quant score <<< "$(variant_spec "$variant")"
    case_dir="$OUT_ROOT/${variant}/repeat_${repeat}"
    output_dir="$case_dir/output"
    run_log="$case_dir/run.log"
    mkdir -p "$output_dir"

    echo
    echo "[gateup-duquant-vs-ose-repeat] repeat=$repeat variant=$variant disable_gateup=$disable_gateup topk=$topk quant=$quant"
    start_ts="$(date +%s)"
    status="ok"
    set +e
    env MODEL="$MODEL" \
      CACHE_DIR="$CACHE_DIR" \
      PLAN_PATH="$PLAN_PATH" \
      OUTPUT_DIR="$output_dir" \
      TEST_DATASET="$TEST_DATASET" \
      TASKS="$TASKS" \
      NUM_FEWSHOT="$NUM_FEWSHOT" \
      BATCH_SIZE="$BATCH_SIZE" \
      NSAMPLES="$NSAMPLES" \
      SEQ_LENGTH="$SEQ_LENGTH" \
      SCALE_SEARCH_STEPS="$SCALE_SEARCH_STEPS" \
      FAST_MOE_CALIB_TOKENS="$FAST_MOE_CALIB_TOKENS" \
      FAST_MOE_DOWN_CALIBRATION="$FAST_MOE_DOWN_CALIBRATION" \
      WEIGHT_CHANNEL_GROUP_SIZE="$WEIGHT_CHANNEL_GROUP_SIZE" \
      ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
      DISABLE_MOE_GATE_UP_DUQUANT_ROTATION="$disable_gateup" \
      MOE_OUTLIER_TOPK="$topk" \
      MOE_OUTLIER_QUANT="$quant" \
      MOE_OUTLIER_SCORE="$score" \
      bash scripts/reproduce_olmoe_691.sh 2>&1 | tee "$run_log"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" != "0" ]]; then
      status="run_failed_${rc}"
    fi
    end_ts="$(date +%s)"
    elapsed=$((end_ts - start_ts))

    rank_log="$(find "$output_dir" -name 'log_rank0_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
    wikitext2="$(extract_last_value 'INFO wikitext2[[:space:]]*:' "$rank_log")"
    c4="$(extract_last_value 'INFO c4[[:space:]]*:' "$rank_log")"
    smooth="$(extract_last_value '\[timing\] smooth_seconds:' "$rank_log")"
    duquant="$(extract_last_value '\[timing\] duquant_seconds:' "$rank_log")"
    calibration="$(extract_last_value '\[timing\] calibration_seconds:' "$rank_log")"
    infer_wiki="$(extract_last_value '\[timing\] inference_seconds\.wikitext2:' "$rank_log")"
    infer_c4="$(extract_last_value '\[timing\] inference_seconds\.c4:' "$rank_log")"
    append_summary "$variant" "$repeat" "$disable_gateup" "$topk" "$quant" "$score" "$status" \
      "$wikitext2" "$c4" "$smooth" "$duquant" "$calibration" "$infer_wiki" "$infer_c4" "$elapsed" "$output_dir" "$run_log" "$rank_log"

    echo "[gateup-duquant-vs-ose-repeat] done variant=$variant repeat=$repeat status=$status elapsed=${elapsed}s"
    echo "[gateup-duquant-vs-ose-repeat] summary: $SUMMARY_CSV"
  done
done

python3 - "$SUMMARY_CSV" "$OUT_ROOT/summary_stats.csv" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, newline="")))
metrics = [
    "wikitext2",
    "c4",
    "smooth_seconds",
    "duquant_seconds",
    "calibration_seconds",
    "inference_wikitext2_seconds",
    "inference_c4_seconds",
    "elapsed_seconds",
]
groups = defaultdict(list)
for row in rows:
    if row["status"] == "ok":
        groups[row["variant"]].append(row)

fieldnames = ["variant", "ok_repeats"]
for metric in metrics:
    fieldnames.extend([f"{metric}_median", f"{metric}_mean", f"{metric}_std"])

with open(dst, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for variant, items in sorted(groups.items()):
        out = {"variant": variant, "ok_repeats": len(items)}
        for metric in metrics:
            vals = []
            for item in items:
                val = item.get(metric, "")
                if val:
                    vals.append(float(val))
            if vals:
                out[f"{metric}_median"] = f"{statistics.median(vals):.6f}"
                out[f"{metric}_mean"] = f"{statistics.mean(vals):.6f}"
                out[f"{metric}_std"] = f"{statistics.stdev(vals):.6f}" if len(vals) > 1 else "0.000000"
            else:
                out[f"{metric}_median"] = ""
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
        writer.writerow(out)
print(f"[gateup-duquant-vs-ose-repeat] stats: {dst}")
PY

echo "[gateup-duquant-vs-ose-repeat] all done"
echo "[gateup-duquant-vs-ose-repeat] detail: $SUMMARY_CSV"
echo "[gateup-duquant-vs-ose-repeat] stats: $OUT_ROOT/summary_stats.csv"
