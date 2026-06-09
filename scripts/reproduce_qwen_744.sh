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
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/repro_qwen_744_$(date +%Y%m%d_%H%M%S)/heteroquant_a8w4}"
TEST_DATASET="${TEST_DATASET-wikitext2,c4}"
TASKS="${TASKS-piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NSAMPLES="${NSAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
FAST_MOE_CALIB_TOKENS="${FAST_MOE_CALIB_TOKENS:-512}"

cd "$ROOT"

task_args=()
if [[ -n "$TASKS" ]]; then
  task_args=(--tasks "$TASKS" --num_fewshot "$NUM_FEWSHOT" --batch_size "$BATCH_SIZE")
fi

python3 main.py \
  --model "$MODEL" \
  --net qwen2_moe \
  --wbits 4 --abits 8 \
  --router_wbits 8 --router_abits 8 \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --w_dynamic_method per_channel_tensor \
  --router_w_dynamic_method per_channel_kl_top0 \
  --nsamples "$NSAMPLES" \
  --seq_length "$SEQ_LENGTH" \
  --eval_ppl \
  --test_dataset "$TEST_DATASET" \
  "${task_args[@]}" \
  --group_size -1 \
  --act_group_size -1 \
  --smooth \
  --fc1_scale_merge act_p99 \
  --act_mean_beta 1 \
  --moe_down_smooth_mode otsu \
  --moe_outlier_topk 64 \
  --moe_outlier_quant w8a8 \
  --disable_moe_gate_up_duquant_rotation \
  --moe_quant_plan "$PLAN_PATH" \
  --moe_outlier_shared_layer_score smooth_scale \
  --fast_moe_down_calibration \
  --fast_moe_calib_tokens "$FAST_MOE_CALIB_TOKENS"
