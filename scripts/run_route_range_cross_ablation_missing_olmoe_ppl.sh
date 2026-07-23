#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Complete the cross-shaped ablation:
#   1) group_size=1, topK sweep: already done in the 20260721_233425 run.
#   2) topK=64, group-size sweep: this script runs only the missing large groups.
export GROUP_SIZES="${GROUP_SIZES:-256,512,1024,2048}"
export TOPKS="${TOPKS:-64}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/route_range_topk64_group_missing_olmoe_ppl_${RUN_ID}}"

exec bash "$ROOT/scripts/run_route_range_topk_group_ablation_olmoe_ppl.sh"
