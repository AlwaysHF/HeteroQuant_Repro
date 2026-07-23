#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-module_ablation_g1}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new -d -s "$SESSION" "cd '$ROOT' && bash scripts/run_module_ablation_route_range_group1_olmoe_ppl.sh"
echo "started tmux session: $SESSION"
echo "attach with: tmux attach -t $SESSION"
