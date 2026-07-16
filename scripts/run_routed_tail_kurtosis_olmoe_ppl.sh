#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/routed_tail_kurtosis_olmoe_ppl_${RUN_ID}}"

env \
  OUT_ROOT="$OUT_ROOT" \
  METRICS="routed_tail,routed_kurtosis" \
  bash scripts/run_metric_ablation_olmoe_ppl.sh
