#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export AW_LABEL="${AW_LABEL:-aw10}"
export TARGET_AVG_SUM_BITS="${TARGET_AVG_SUM_BITS:-10.0}"
export TIER_QUANTS="${TIER_QUANTS:-w4g-1-a4g-1,w5g-1-a8g-1,w6g-1-a8g-1}"
# A4W4/A8W5/A8W6 total bits are 8/13/14; 0.625/0.25/0.125 averages to 10.0.
export TIER_FRACTIONS="${TIER_FRACTIONS:-0.625,0.25,0.125}"
export OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/route_range_aw10_g1_g2048_olmoe_qwen_ppl_downstream_${RUN_ID}}"

exec bash "$ROOT/scripts/run_route_range_aw11_g1_g2048_olmoe_qwen_ppl_downstream.sh"
