#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-gateup_e2e_fixed}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPEATS="${REPEATS:-3}"
TEST_DATASET="${TEST_DATASET:-wikitext2,c4}"
TASKS="${TASKS:-}"
WEIGHT_CHANNEL_GROUP_SIZE="${WEIGHT_CHANNEL_GROUP_SIZE:-2048}"
ROUTER_WEIGHT_CHANNEL_GROUP_SIZE="${ROUTER_WEIGHT_CHANNEL_GROUP_SIZE:-$WEIGHT_CHANNEL_GROUP_SIZE}"
VARIANTS="${VARIANTS:-full_duquant,no_gateup_no_ose,ours_ose_w8a8,ours_ose_fp16}"
RUN_ID="${RUN_ID:-fixed_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/gateup_duquant_vs_ose_repeated_olmoe_${RUN_ID}}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[start-gateup-e2e] tmux session already exists: $SESSION"
  echo "[start-gateup-e2e] attach: tmux attach -t $SESSION"
  echo "[start-gateup-e2e] or choose another session: SESSION=${SESSION}_2 bash $0"
  exit 1
fi

mkdir -p "$OUT_ROOT"

CMD=$(
  printf 'cd %q && env CUDA_VISIBLE_DEVICES=%q REPEATS=%q TEST_DATASET=%q TASKS=%q WEIGHT_CHANNEL_GROUP_SIZE=%q ROUTER_WEIGHT_CHANNEL_GROUP_SIZE=%q VARIANTS=%q RUN_ID=%q OUT_ROOT=%q bash scripts/run_gateup_duquant_vs_ose_repeated_olmoe.sh' \
    "$ROOT" \
    "$CUDA_VISIBLE_DEVICES" \
    "$REPEATS" \
    "$TEST_DATASET" \
    "$TASKS" \
    "$WEIGHT_CHANNEL_GROUP_SIZE" \
    "$ROUTER_WEIGHT_CHANNEL_GROUP_SIZE" \
    "$VARIANTS" \
    "$RUN_ID" \
    "$OUT_ROOT"
)

echo "[start-gateup-e2e] session: $SESSION"
echo "[start-gateup-e2e] gpu: $CUDA_VISIBLE_DEVICES"
echo "[start-gateup-e2e] repeats: $REPEATS"
echo "[start-gateup-e2e] datasets: $TEST_DATASET"
echo "[start-gateup-e2e] tasks: ${TASKS:-<none>}"
echo "[start-gateup-e2e] variants: $VARIANTS"
echo "[start-gateup-e2e] output: $OUT_ROOT"
echo "[start-gateup-e2e] command log: $OUT_ROOT/start_command.txt"
printf '%s\n' "$CMD" > "$OUT_ROOT/start_command.txt"

tmux new -d -s "$SESSION" "$CMD"

echo "[start-gateup-e2e] started"
echo "[start-gateup-e2e] attach: tmux attach -t $SESSION"
echo "[start-gateup-e2e] watch summary: tail -f $OUT_ROOT/summary.csv"
