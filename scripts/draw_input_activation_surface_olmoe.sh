#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

MODEL="${MODEL:-$ROOT/local_models/olmoe_compat}"
DATASET_PATH="${DATASET_PATH:-/cephfs/shared/lwk/notebook04/datasets/wiki2_raw_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/figures/input_activation_surface}"
LAYER="${LAYER:-0}"
EXPERTS="${EXPERTS:-6,15,60}"
TOKENS_PER_EXPERT="${TOKENS_PER_EXPERT:-200}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MAX_CHUNKS="${MAX_CHUNKS:-4}"
CHANNEL_STRIDE="${CHANNEL_STRIDE:-4}"
CLIP_PERCENTILE="${CLIP_PERCENTILE:-99.6}"
DPI="${DPI:-220}"

mkdir -p "$OUTPUT_DIR"

python3 tools/input_activation_channel_surface.py \
  --model "$MODEL" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --layer "$LAYER" \
  --experts "$EXPERTS" \
  --tokens_per_expert "$TOKENS_PER_EXPERT" \
  --seq_length "$SEQ_LENGTH" \
  --max_chunks "$MAX_CHUNKS" \
  --channel_stride "$CHANNEL_STRIDE" \
  --clip_percentile "$CLIP_PERCENTILE" \
  --dpi "$DPI"
