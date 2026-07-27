#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/post_quant_moe_ffn_group_sweep_${RUN_ID}}"

GROUP_SIZES="${GROUP_SIZES:-1,2,4,8,16,32,64,128,256,512,1024,2048}"
PRESETS="${PRESETS:-olmoe}"
SCOPE="${SCOPE:-expert_ffn}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-1,8,16,64}"
TOKENS_PER_EXPERT="${TOKENS_PER_EXPERT:-1,2,4,8,16,32}"
TOTAL_TOKENS="${TOTAL_TOKENS:-0}"
VARIANTS="${VARIANTS:-duquant_once,plain_no_ose,ours_ose_w8a8}"
OSE_WEIGHT_GROUP_SIZE="${OSE_WEIGHT_GROUP_SIZE:-1}"
VALIDATE="${VALIDATE:-0}"
WARMUP="${WARMUP:-20}"
REPEAT="${REPEAT:-80}"
ROUNDS="${ROUNDS:-5}"

mkdir -p "$OUT_ROOT"

echo "[group-sweep] output: $OUT_ROOT"
echo "[group-sweep] group_sizes: $GROUP_SIZES"
echo "[group-sweep] presets: $PRESETS"
echo "[group-sweep] scope: $SCOPE"
echo "[group-sweep] active_experts: $ACTIVE_EXPERTS"
echo "[group-sweep] tokens_per_expert: $TOKENS_PER_EXPERT"
echo "[group-sweep] total_tokens: $TOTAL_TOKENS"
echo "[group-sweep] variants: $VARIANTS"

IFS=',' read -r -a group_array <<< "$GROUP_SIZES"
for group_size in "${group_array[@]}"; do
  group_size="$(echo "$group_size" | xargs)"
  [[ -n "$group_size" ]] || continue
  group_dir="$OUT_ROOT/group_${group_size}"
  echo "[group-sweep] running MAIN_WEIGHT_GROUP_SIZE=$group_size -> $group_dir"
  OUT_ROOT="$group_dir" \
    PRESETS="$PRESETS" \
    SCOPE="$SCOPE" \
    ACTIVE_EXPERTS="$ACTIVE_EXPERTS" \
    TOKENS_PER_EXPERT="$TOKENS_PER_EXPERT" \
    TOTAL_TOKENS="$TOTAL_TOKENS" \
    VARIANTS="$VARIANTS" \
    MAIN_WEIGHT_GROUP_SIZE="$group_size" \
    OSE_WEIGHT_GROUP_SIZE="$OSE_WEIGHT_GROUP_SIZE" \
    VALIDATE="$VALIDATE" \
    WARMUP="$WARMUP" \
    REPEAT="$REPEAT" \
    ROUNDS="$ROUNDS" \
    bash scripts/run_post_quant_moe_ffn_duquant_vs_ose_benchmark.sh
done

python3 - "$OUT_ROOT" <<'PY'
import csv
import os
import statistics
import sys

root = sys.argv[1]
rows = []
for name in sorted(os.listdir(root)):
    if not name.startswith("group_"):
        continue
    group = int(name.split("_", 1)[1])
    path = os.path.join(root, name, "post_quant_moe_ffn_duquant_vs_ose_detail.csv")
    if not os.path.exists(path):
        continue
    detail = list(csv.DictReader(open(path)))
    ours = [r for r in detail if r["variant"] == "ours_ose_w8a8" and r["speedup_over_duquant_once"]]
    if not ours:
        continue
    speedups = [float(r["speedup_over_duquant_once"]) for r in ours]
    row = {
        "main_weight_group_size": group,
        "num_cases": len(ours),
        "speedup_min": f"{min(speedups):.6f}",
        "speedup_median": f"{statistics.median(speedups):.6f}",
        "speedup_mean": f"{statistics.mean(speedups):.6f}",
        "speedup_max": f"{max(speedups):.6f}",
    }
    for e in sorted({int(r["active_experts"]) for r in ours}):
        vals = [float(r["speedup_over_duquant_once"]) for r in ours if int(r["active_experts"]) == e]
        row[f"E{e}_speedup_median"] = f"{statistics.median(vals):.6f}"
    rows.append(row)

fields = []
for row in rows:
    for key in row:
        if key not in fields:
            fields.append(key)

out = os.path.join(root, "group_sweep_summary.csv")
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"[group-sweep] summary: {out}")
PY

echo "[group-sweep] done"
echo "[group-sweep] summary: $OUT_ROOT/group_sweep_summary.csv"
