#!/usr/bin/env bash
set -Eeuo pipefail
cd "/home/lwk/HeteroQuant_Repro"
echo "[suite] start: $(date '+%F %T')"
echo "[suite] output root: /home/lwk/HeteroQuant_Repro/experiments/remaining_ablation_suite_20260716_210440"

echo
echo "[suite] 1/3 component ablation"
CUDA_VISIBLE_DEVICES="0" OUT_ROOT="/home/lwk/HeteroQuant_Repro/experiments/remaining_ablation_suite_20260716_210440/component" \
  bash scripts/run_component_ablation_routed_tail_olmoe_ppl_group2048.sh

echo
echo "[suite] 2/3 OSE precision ablation"
CUDA_VISIBLE_DEVICES="0" OUT_ROOT="/home/lwk/HeteroQuant_Repro/experiments/remaining_ablation_suite_20260716_210440/ose_precision" \
  bash scripts/run_ose_precision_routed_tail_olmoe_ppl_group2048.sh

echo
echo "[suite] 3/3 OSE channel-score ablation"
CUDA_VISIBLE_DEVICES="0" OUT_ROOT="/home/lwk/HeteroQuant_Repro/experiments/remaining_ablation_suite_20260716_210440/ose_score" \
  bash scripts/run_ose_score_routed_tail_olmoe_ppl_group2048.sh

echo
echo "[suite] done: $(date '+%F %T')"
echo "[suite] summaries:"
echo "/home/lwk/HeteroQuant_Repro/experiments/remaining_ablation_suite_20260716_210440/component/summary.csv"
echo "/home/lwk/HeteroQuant_Repro/experiments/remaining_ablation_suite_20260716_210440/ose_precision/summary.csv"
echo "/home/lwk/HeteroQuant_Repro/experiments/remaining_ablation_suite_20260716_210440/ose_score/summary.csv"
