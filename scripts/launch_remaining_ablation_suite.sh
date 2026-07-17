#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-remaining_ablation_2048}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/remaining_ablation_suite_${RUN_ID}}"
PARALLEL="${PARALLEL:-0}"

mkdir -p "$OUT_ROOT"

COMPONENT_OUT="${COMPONENT_OUT:-$OUT_ROOT/component}"
PRECISION_OUT="${PRECISION_OUT:-$OUT_ROOT/ose_precision}"
SCORE_OUT="${SCORE_OUT:-$OUT_ROOT/ose_score}"

COMPONENT_CUDA_VISIBLE_DEVICES="${COMPONENT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
PRECISION_CUDA_VISIBLE_DEVICES="${PRECISION_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
SCORE_CUDA_VISIBLE_DEVICES="${SCORE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

write_sequential_runner() {
  local runner="$OUT_ROOT/run_all_sequential.sh"
  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd "$ROOT"
echo "[suite] start: \$(date '+%F %T')"
echo "[suite] output root: $OUT_ROOT"

echo
echo "[suite] 1/3 component ablation"
CUDA_VISIBLE_DEVICES="$COMPONENT_CUDA_VISIBLE_DEVICES" OUT_ROOT="$COMPONENT_OUT" \\
  bash scripts/run_component_ablation_routed_tail_olmoe_ppl_group2048.sh

echo
echo "[suite] 2/3 OSE precision ablation"
CUDA_VISIBLE_DEVICES="$PRECISION_CUDA_VISIBLE_DEVICES" OUT_ROOT="$PRECISION_OUT" \\
  bash scripts/run_ose_precision_routed_tail_olmoe_ppl_group2048.sh

echo
echo "[suite] 3/3 OSE channel-score ablation"
CUDA_VISIBLE_DEVICES="$SCORE_CUDA_VISIBLE_DEVICES" OUT_ROOT="$SCORE_OUT" \\
  bash scripts/run_ose_score_routed_tail_olmoe_ppl_group2048.sh

echo
echo "[suite] done: \$(date '+%F %T')"
echo "[suite] summaries:"
echo "$COMPONENT_OUT/summary.csv"
echo "$PRECISION_OUT/summary.csv"
echo "$SCORE_OUT/summary.csv"
EOF
  chmod +x "$runner"
  echo "$runner"
}

write_parallel_runner() {
  local name="$1"
  local cuda="$2"
  local out="$3"
  local script="$4"
  local runner="$OUT_ROOT/run_${name}.sh"
  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd "$ROOT"
echo "[$name] start: \$(date '+%F %T')"
CUDA_VISIBLE_DEVICES="$cuda" OUT_ROOT="$out" bash "$script"
echo "[$name] done: \$(date '+%F %T')"
echo "[$name] summary: $out/summary.csv"
EOF
  chmod +x "$runner"
  echo "$runner"
}

if [[ "$PARALLEL" == "1" || "$PARALLEL" == "true" || "$PARALLEL" == "TRUE" ]]; then
  component_runner="$(write_parallel_runner component "$COMPONENT_CUDA_VISIBLE_DEVICES" "$COMPONENT_OUT" scripts/run_component_ablation_routed_tail_olmoe_ppl_group2048.sh)"
  precision_runner="$(write_parallel_runner ose_precision "$PRECISION_CUDA_VISIBLE_DEVICES" "$PRECISION_OUT" scripts/run_ose_precision_routed_tail_olmoe_ppl_group2048.sh)"
  score_runner="$(write_parallel_runner ose_score "$SCORE_CUDA_VISIBLE_DEVICES" "$SCORE_OUT" scripts/run_ose_score_routed_tail_olmoe_ppl_group2048.sh)"

  tmux new-session -d -s "$SESSION" -n component "$component_runner"
  tmux new-window -t "$SESSION" -n ose_precision "$precision_runner"
  tmux new-window -t "$SESSION" -n ose_score "$score_runner"
else
  sequential_runner="$(write_sequential_runner)"
  tmux new-session -d -s "$SESSION" -n suite "$sequential_runner"
fi

echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "output root: $OUT_ROOT"
echo "component summary: $COMPONENT_OUT/summary.csv"
echo "OSE precision summary: $PRECISION_OUT/summary.csv"
echo "OSE score summary: $SCORE_OUT/summary.csv"
