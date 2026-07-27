#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/post_quant_moe_ffn_ose_topk_sweep_${RUN_ID}}"

OSE_TOPK_VALUES="${OSE_TOPK_VALUES:-${K_VALUES:-16,32,64,128,256}}"
PRESETS="${PRESETS:-olmoe}"
SCOPE="${SCOPE:-routed_moe_layer}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-1,8,16,64}"
TOKENS_PER_EXPERT="${TOKENS_PER_EXPERT:-1,2,4,8,16,32}"
TOTAL_TOKENS="${TOTAL_TOKENS:-0}"
VARIANTS="${VARIANTS:-duquant_once,plain_no_ose,ours_ose_w8a8}"

MAIN_WEIGHT_GROUP_SIZE="${MAIN_WEIGHT_GROUP_SIZE:-2048}"
OSE_WEIGHT_GROUP_SIZE="${OSE_WEIGHT_GROUP_SIZE:-1}"
VALIDATE="${VALIDATE:-0}"
WARMUP="${WARMUP:-20}"
REPEAT="${REPEAT:-80}"
ROUNDS="${ROUNDS:-5}"

mkdir -p "$OUT_ROOT"

echo "[ose-topk-sweep] output: $OUT_ROOT"
echo "[ose-topk-sweep] ose_topk_values: $OSE_TOPK_VALUES"
echo "[ose-topk-sweep] presets: $PRESETS"
echo "[ose-topk-sweep] scope: $SCOPE"
echo "[ose-topk-sweep] active_experts: $ACTIVE_EXPERTS"
echo "[ose-topk-sweep] tokens_per_expert: $TOKENS_PER_EXPERT"
echo "[ose-topk-sweep] total_tokens: $TOTAL_TOKENS"
echo "[ose-topk-sweep] main_weight_group_size=$MAIN_WEIGHT_GROUP_SIZE ose_weight_group_size=$OSE_WEIGHT_GROUP_SIZE"

IFS=',' read -r -a topk_array <<< "$OSE_TOPK_VALUES"
for topk in "${topk_array[@]}"; do
  topk="$(echo "$topk" | xargs)"
  [[ -n "$topk" ]] || continue
  topk_dir="$OUT_ROOT/k_${topk}"
  echo "[ose-topk-sweep] running OSE_TOPK=$topk -> $topk_dir"
  OUT_ROOT="$topk_dir" \
    PRESETS="$PRESETS" \
    SCOPE="$SCOPE" \
    ACTIVE_EXPERTS="$ACTIVE_EXPERTS" \
    TOKENS_PER_EXPERT="$TOKENS_PER_EXPERT" \
    TOTAL_TOKENS="$TOTAL_TOKENS" \
    VARIANTS="$VARIANTS" \
    OSE_TOPK="$topk" \
    MAIN_WEIGHT_GROUP_SIZE="$MAIN_WEIGHT_GROUP_SIZE" \
    OSE_WEIGHT_GROUP_SIZE="$OSE_WEIGHT_GROUP_SIZE" \
    VALIDATE="$VALIDATE" \
    WARMUP="$WARMUP" \
    REPEAT="$REPEAT" \
    ROUNDS="$ROUNDS" \
    bash scripts/run_post_quant_moe_ffn_duquant_vs_ose_benchmark.sh

  python3 tools/post_quant_ffn_compute_metrics.py \
    --input_csv "$topk_dir/post_quant_moe_ffn_duquant_vs_ose_detail.csv" \
    --output_csv "$topk_dir/theoretical_metrics_detail.csv" \
    --summary_csv "$topk_dir/theoretical_metrics_summary.csv"
done

python3 - "$OUT_ROOT" <<'PY'
import csv
import os
import statistics
import sys

root = sys.argv[1]
rows = []
case_rows = []
for name in sorted(os.listdir(root), key=lambda value: int(value.split("_", 1)[1]) if value.startswith("k_") and value.split("_", 1)[1].isdigit() else -1):
    if not name.startswith("k_"):
        continue
    topk = int(name.split("_", 1)[1])
    detail_path = os.path.join(root, name, "post_quant_moe_ffn_duquant_vs_ose_detail.csv")
    metrics_path = os.path.join(root, name, "theoretical_metrics_detail.csv")
    if not os.path.exists(detail_path):
        continue
    detail = list(csv.DictReader(open(detail_path)))
    metrics = list(csv.DictReader(open(metrics_path))) if os.path.exists(metrics_path) else []
    ours = [r for r in detail if r["variant"] == "ours_ose_w8a8" and r["speedup_over_duquant_once"]]
    if not ours:
        continue
    speedups = [float(r["speedup_over_duquant_once"]) for r in ours]
    row = {
        "ose_topk": topk,
        "num_cases": len(ours),
        "speedup_min": f"{min(speedups):.6f}",
        "speedup_median": f"{statistics.median(speedups):.6f}",
        "speedup_mean": f"{statistics.mean(speedups):.6f}",
        "speedup_max": f"{max(speedups):.6f}",
    }
    for e in sorted({int(r["active_experts"]) for r in ours}):
        vals = [float(r["speedup_over_duquant_once"]) for r in ours if int(r["active_experts"]) == e]
        row[f"E{e}_speedup_median"] = f"{statistics.median(vals):.6f}"
        row[f"E{e}_fast_cases"] = sum(v > 1.0 for v in vals)
    metric_ours = [r for r in metrics if r.get("variant") == "ours_ose_w8a8"]
    if metric_ours:
        row["ose_branch_macs_per_routed_token"] = metric_ours[0]["ose_unique_overhead_macs_per_routed_token"]
        row["duquant_gateup_macs_per_routed_token"] = metric_ours[0]["duquant_unique_overhead_macs_per_routed_token"]
        row["duquant_over_ose_macs"] = metric_ours[0]["unique_overhead_reduction_duquant_vs_ose"]
    rows.append(row)

    by_key = {
        (int(r["active_experts"]), int(r["tokens_per_expert"]), r["variant"]): r
        for r in detail
    }
    keys = sorted({(int(r["active_experts"]), int(r["tokens_per_expert"])) for r in detail})
    for e, t in keys:
        ours_row = by_key.get((e, t, "ours_ose_w8a8"))
        du_row = by_key.get((e, t, "duquant_once"))
        plain_row = by_key.get((e, t, "plain_no_ose"))
        if not ours_row or not du_row:
            continue
        case_rows.append(
            {
                "ose_topk": topk,
                "active_experts": e,
                "tokens_per_expert": t,
                "total_tokens": ours_row.get("total_tokens", ""),
                "duquant_once_ms": du_row["latency_ms"],
                "plain_no_ose_ms": plain_row["latency_ms"] if plain_row else "",
                "ours_ose_w8a8_ms": ours_row["latency_ms"],
                "speedup_over_duquant_once": ours_row["speedup_over_duquant_once"],
                "ours_latency_std_ms": ours_row["latency_std_ms"],
            }
        )

fields = []
for row in rows:
    for key in row:
        if key not in fields:
            fields.append(key)
summary_path = os.path.join(root, "ose_topk_sweep_summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

case_fields = list(case_rows[0].keys()) if case_rows else []
case_path = os.path.join(root, "ose_topk_sweep_cases.csv")
with open(case_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=case_fields)
    writer.writeheader()
    writer.writerows(case_rows)

md_path = os.path.join(root, "ose_topk_sweep_summary.md")
with open(md_path, "w") as f:
    f.write("# OSE Top-k Sweep Summary\n\n")
    f.write(f"Root: `{root}`\n\n")
    f.write("Speedup is `duquant_once_latency / ours_ose_w8a8_latency`; larger is better for ours.\n\n")
    if rows:
        display_fields = [
            "ose_topk",
            "speedup_median",
            "speedup_mean",
            "E1_speedup_median",
            "E8_speedup_median",
            "E16_speedup_median",
            "E64_speedup_median",
            "duquant_over_ose_macs",
        ]
        f.write("| " + " | ".join(display_fields) + " |\n")
        f.write("|" + "|".join(["---:" for _ in display_fields]) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(field, "")) for field in display_fields) + " |\n")

print(f"[ose-topk-sweep] summary: {summary_path}")
print(f"[ose-topk-sweep] cases: {case_path}")
print(f"[ose-topk-sweep] markdown: {md_path}")
PY

echo "[ose-topk-sweep] done"
echo "[ose-topk-sweep] summary: $OUT_ROOT/ose_topk_sweep_summary.csv"
echo "[ose-topk-sweep] cases: $OUT_ROOT/ose_topk_sweep_cases.csv"
