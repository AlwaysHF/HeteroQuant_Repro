#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

MODEL="${MODEL:-$ROOT/local_models/Qwen1.5-MoE-A2.7B}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
PLAN_PATH="${PLAN_PATH:-$ROOT/plans/qwen_routed_range/moe_quant_plan_routed_range.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/qwen_shared_expert_migration_diag_$(date +%Y%m%d_%H%M%S)}"
CALIB_DATASET="${CALIB_DATASET:-wikitext2}"
NSAMPLES="${NSAMPLES:-128}"
FUNCTIONAL_NSAMPLES="${FUNCTIONAL_NSAMPLES:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MAX_GATE_TOKENS="${MAX_GATE_TOKENS:-1024}"
MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
ROUTED_EXPERT_SAMPLE="${ROUTED_EXPERT_SAMPLE:-4}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

python3 tools/qwen_shared_expert_migration_diag.py \
  --model "$MODEL" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --moe_quant_plan "$PLAN_PATH" \
  --calib_dataset "$CALIB_DATASET" \
  --nsamples "$NSAMPLES" \
  --functional_nsamples "$FUNCTIONAL_NSAMPLES" \
  --seq_length "$SEQ_LENGTH" \
  --max_gate_tokens "$MAX_GATE_TOKENS" \
  --moe_outlier_topk "$MOE_OUTLIER_TOPK" \
  --routed_expert_sample "$ROUTED_EXPERT_SAMPLE" \
  2>&1 | tee "$OUTPUT_DIR/console.log"
