#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUQUANT_TORCH_NUM_THREADS="${DUQUANT_TORCH_NUM_THREADS:-8}"
export DUQUANT_TORCH_NUM_INTEROP_THREADS="${DUQUANT_TORCH_NUM_INTEROP_THREADS:-8}"
export LM_EVAL_LOCAL_DATASET_ROOT="${LM_EVAL_LOCAL_DATASET_ROOT:-/cephfs/shared/lwk/notebook04/datasets}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-$ROOT/experiments/fp16_and_qwen_group2048_${RUN_ID}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"

OLMOE_MODEL="${OLMOE_MODEL:-$ROOT/local_models/olmoe_compat}"
QWEN_MODEL="${QWEN_MODEL:-$ROOT/local_models/Qwen1.5-MoE-A2.7B}"

TEST_DATASET="${TEST_DATASET-wikitext2,c4}"
TASKS="${TASKS-piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"

NSAMPLES="${NSAMPLES:-128}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"
MOE_OUTLIER_TOPK="${MOE_OUTLIER_TOPK:-64}"
MOE_OUTLIER_QUANT="${MOE_OUTLIER_QUANT:-w8a8}"
MOE_OUTLIER_SCORE="${MOE_OUTLIER_SCORE:-smooth_scale}"
DISABLE_MOE_GATE_UP_DUQUANT_ROTATION="${DISABLE_MOE_GATE_UP_DUQUANT_ROTATION:-1}"

mkdir -p "$EXP_ROOT/logs"

run_and_log() {
  local name="$1"
  shift
  local log_file="$EXP_ROOT/logs/${name}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] start ${name}"
  echo "log: ${log_file}"
  "$@" 2>&1 | tee "$log_file"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] done ${name}"
}

common_fp16_args=(
  --cache_dir "$CACHE_DIR"
  --eval_ppl
  --test_dataset "$TEST_DATASET"
  --tasks "$TASKS"
  --num_fewshot "$NUM_FEWSHOT"
  --batch_size "$BATCH_SIZE"
  --seq_length "$SEQ_LENGTH"
  --wbits 16
  --abits 16
  --router_wbits 16
  --router_abits 16
  --smooth
)

run_and_log fp16_olmoe \
  "$PYTHON_BIN" main.py \
    --model "$OLMOE_MODEL" \
    --model_name olmoe \
    --output_dir "$EXP_ROOT/fp16_olmoe" \
    "${common_fp16_args[@]}"

run_and_log fp16_qwen \
  "$PYTHON_BIN" main.py \
    --model "$QWEN_MODEL" \
    --model_name qwen2_moe \
    --output_dir "$EXP_ROOT/fp16_qwen" \
    "${common_fp16_args[@]}"

run_and_log qwen_group2048 \
  env \
    MODEL="$QWEN_MODEL" \
    CACHE_DIR="$CACHE_DIR" \
    OUTPUT_DIR="$EXP_ROOT/qwen_group2048" \
    TEST_DATASET="$TEST_DATASET" \
    TASKS="$TASKS" \
    NUM_FEWSHOT="$NUM_FEWSHOT" \
    BATCH_SIZE="$BATCH_SIZE" \
    NSAMPLES="$NSAMPLES" \
    SEQ_LENGTH="$SEQ_LENGTH" \
    FAST_MOE_CALIB_TOKENS="$FAST_MOE_CALIB_TOKENS" \
    MOE_OUTLIER_TOPK="$MOE_OUTLIER_TOPK" \
    MOE_OUTLIER_QUANT="$MOE_OUTLIER_QUANT" \
    MOE_OUTLIER_SCORE="$MOE_OUTLIER_SCORE" \
    DISABLE_MOE_GATE_UP_DUQUANT_ROTATION="$DISABLE_MOE_GATE_UP_DUQUANT_ROTATION" \
    WEIGHT_CHANNEL_GROUP_SIZE=2048 \
    ROUTER_WEIGHT_CHANNEL_GROUP_SIZE=2048 \
    bash scripts/reproduce_qwen_744.sh

echo "All experiments finished. Outputs are under: $EXP_ROOT"
