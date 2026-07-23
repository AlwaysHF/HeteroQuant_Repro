#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-ose_shared_scale_only}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new -d -s "$SESSION" "cd '$ROOT' && SELECTION_SCORES=shared_scale bash scripts/run_ose_selection_metric_ablation_olmoe_ppl.sh"
echo "started tmux session: $SESSION"
echo "attach with: tmux attach -t $SESSION"
