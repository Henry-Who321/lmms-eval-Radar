#!/bin/bash
set -u

# Qwen3-VL Batch Evaluation Script with SGLang Backend
# This script evaluates multiple checkpoints sequentially
#
# Requirements:
# - sglang>=0.4.6
# - qwen-vl-utils
# - CUDA-enabled GPU(s)

# ============================================================================
# Configuration
# ============================================================================
    # "/mnt/publicdataset/Qwen/Qwen3-VL-4B-Instruct"

# Parallelization Settings
TENSOR_PARALLEL_SIZE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Memory and Performance Settings
GPU_MEMORY_UTILIZATION=0.75
BATCH_SIZE=1

# SGLang Specific Settings
MAX_PIXELS=1605632
MIN_PIXELS=784
MAX_FRAME_NUM=32
THREADS=16

declare -a MODELS=(
#----------
"/vlm/chenfurui/AReaL/tmp/experiments_stage2fr_0402/checkpoints/root/stage2_gspo_rollout_32_bs30_v2_cos_fr0402/trial_002/default/epoch0epochstep104globalstep104"
)

# Generation Configuration
GEN_KWARGS="max_new_tokens=4096,until="
TASKS="mme,mmstar,chartqa,realworldqa,mathvista_testmini_solution,mathverse_testmini,ai2d2"
# Base Output Configurations
OUTPUT_PATH="/vlm/chenfurui/lmms-eval/results/"
LOG_SAMPLES=true
LOG_SUFFIX="qwen3vl_auto"
SUMMARY_LOG="${OUTPUT_PATH}/batch_summary.log"

# Environment Variables
export LMMS_EXTRACT_ANSWER_FROM_TAGS=true
export LMMS_TEST_MODE=stage2

# Reduce the chance of one TP worker crashing and taking down the whole scheduler.
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# Batch Evaluation Loop
# ============================================================================

# Create output base directory
mkdir -p "$OUTPUT_PATH"
echo "Batch Evaluation Summary - $(date)" > "$SUMMARY_LOG"
echo "===========================================" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

# Counters
TOTAL_MODELS=${#MODELS[@]}
SUCCESS_COUNT=0
FAILED_COUNT=0
declare -a FAILED_MODELS=()

echo "=========================================="
echo "Batch Evaluation Started"
echo "=========================================="
echo "Total Models to Evaluate: $TOTAL_MODELS"
echo "Tasks: $TASKS"
echo "=========================================="
echo ""

# Loop through each model
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NUM=$((i + 1))

    # Extract a short name from the model path for output directory
    # This extracts: trial_X/default/epochXstepYglobalstepZ
    MODEL_SHORT=$(echo "$MODEL" | grep -oP 'trial_\d+/default/[^/]+')
    if [ -z "$MODEL_SHORT" ]; then
        MODEL_SHORT=$(basename "$MODEL")
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating Model $MODEL_NUM/$TOTAL_MODELS"
    echo "=========================================="
    echo "Model: $MODEL"
    echo "Short Name: $MODEL_SHORT"
    echo "Output Path: $OUTPUT_PATH"
    echo "=========================================="

    MODEL_OUTPUT_PATH="${OUTPUT_PATH}/${MODEL_SHORT//\//__}"
    mkdir -p "$MODEL_OUTPUT_PATH"

    # Build the command
    CMD="${PYTHON_BIN} -m lmms_eval \
        --model sglang \
        --model_args model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},max_frame_num=${MAX_FRAME_NUM},threads=${THREADS} \
        --tasks ${TASKS} \
        --batch_size ${BATCH_SIZE} \
        --output_path ${MODEL_OUTPUT_PATH}"

    # Add optional arguments
    if [ "$LOG_SAMPLES" = true ]; then
        CMD="$CMD --log_samples --log_samples_suffix ${LOG_SUFFIX}"
    fi

    if [ -n "${LIMIT:-}" ]; then
        CMD="$CMD --limit ${LIMIT}"
    fi

    # Add generation kwargs to override task defaults (e.g., max_new_tokens)
    if [ -n "${GEN_KWARGS:-}" ]; then
        CMD="$CMD --gen_kwargs ${GEN_KWARGS}"
    fi

    # Execute evaluation
    START_TIME=$(date +%s)
    eval $CMD
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "[$(date)] SUCCESS | ${MODEL_SHORT} | ${DURATION}s" >> "$SUMMARY_LOG"
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("${MODEL_SHORT}")
        echo "[$(date)] FAILED  | ${MODEL_SHORT} | exit=${EXIT_CODE} | ${DURATION}s" >> "$SUMMARY_LOG"
        pkill -f 'sglang|lmms_eval' >/dev/null 2>&1 || true
    fi

    echo "----------------------------------------"

    # Small delay between models to allow GPU memory cleanup
    if [ $MODEL_NUM -lt $TOTAL_MODELS ]; then
        echo "Waiting 30 seconds before next model..."
        sleep 30
    fi
done

echo "" | tee -a "$SUMMARY_LOG"
echo "Batch finished: success=${SUCCESS_COUNT}, failed=${FAILED_COUNT}" | tee -a "$SUMMARY_LOG"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed models:" | tee -a "$SUMMARY_LOG"
    printf '  %s\n' "${FAILED_MODELS[@]}" | tee -a "$SUMMARY_LOG"
fi
