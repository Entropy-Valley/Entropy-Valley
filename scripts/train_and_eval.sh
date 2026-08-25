#!/bin/bash
# LaDiT: train (LoRA SFT) + evaluate (Oracle / Fixed-Ratio / Entropy-Valley).
#
# Usage:
#   bash scripts/train_and_eval.sh <config.yaml> [SEED]
#
# Example:
#   bash scripts/train_and_eval.sh configs/enzh.yaml 42
#   bash scripts/train_and_eval.sh configs/zhen.yaml 42
#   bash scripts/train_and_eval.sh configs/ende.yaml 42
set -euo pipefail
export PYTHONUNBUFFERED=1

CONFIG="${1:?Usage: $0 <config.yaml> [SEED]}"
SEED="${2:-42}"
LANG_PAIR=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['data']['lang_pair'])")
MODEL_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model']['path'])")
TRAIN_DATA=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['data']['train'])")
DEV_DATA=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['data']['dev'])")
TEST_DATA=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['evaluation']['test_sets'][0])")
OUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['checkpointing']['output_dir'])")
BASELINE_RATIO=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['evaluation']['length_ratio_baseline'])")
EV_RATIOS=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['evaluation']['ev_candidate_ratios'])")

RUN_NAME="ladit_${LANG_PAIR//-/}_seed${SEED}"
OUT_DIR="${OUT_DIR}_seed${SEED}"
EVAL_DIR="eval_results/${RUN_NAME}"

echo "=== ${RUN_NAME} | config=${CONFIG} | seed=${SEED} ==="

# Phase 1: LoRA SFT training (skip if checkpoint exists)
if ls ${OUT_DIR}/checkpoint-* 1>/dev/null 2>&1; then
    CKPT=$(ls -d ${OUT_DIR}/checkpoint-* | sort -t- -k2 -n | tail -1)
    echo "Existing checkpoint: ${CKPT}, skipping training."
else
    mkdir -p "${OUT_DIR}" logs
    N_GPU=$(python3 -c "import os; print(len(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')))")
    torchrun --nproc_per_node="${N_GPU}" --master_port=29500 \
        -m ladit.training.train \
        --model_path "${MODEL_PATH}" \
        --use_lora --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
        --gradient_checkpointing \
        --train_data "${TRAIN_DATA}" --dev_data "${DEV_DATA}" \
        --max_seq_len 1024 \
        --noise_schedule uniform \
        --lang_pair "${LANG_PAIR}" \
        --batch_size 4 --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 --weight_decay 0.01 --num_epochs 3 \
        --warmup_ratio 0.05 --lr_scheduler cosine --max_grad_norm 1.0 \
        --seed "${SEED}" --bf16 \
        --output_dir "${OUT_DIR}" \
        --save_steps 500 --eval_steps 500 --logging_steps 10 \
        2>&1 | tee "logs/${RUN_NAME}_train.log"
    CKPT=$(ls -d ${OUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi

# Phase 2: Evaluate Oracle / Ratio / Entropy-Valley on the test set
mkdir -p "${EVAL_DIR}"
NUM_TEST=$(wc -l < "${TEST_DATA}")
echo "=== Decoding ${NUM_TEST} test sentences ==="

python scripts/decode_eval.py \
    --model_path "${MODEL_PATH}" \
    --lora_path "${CKPT}" \
    --input_file "${TEST_DATA}" \
    --output_dir "${EVAL_DIR}" \
    --num_examples "${NUM_TEST}" \
    --num_steps 32 \
    --schedule med \
    --methods "oracle,ratio_${BASELINE_RATIO},entropy_valley" \
    --candidate_ratios "${EV_RATIOS}" \
    --lang_pair "${LANG_PAIR}" \
    --device cuda

# Phase 3: BLEU + COMET (per-method, calls evaluate.py once per translation file)
for METHOD in oracle "ratio_${BASELINE_RATIO}" entropy_valley; do
    python -m ladit.evaluation.evaluate \
        --translations_file "${EVAL_DIR}/translations_${METHOD}.json" \
        --output_file "${EVAL_DIR}/metrics_${METHOD}.json" \
        --comet_model Unbabel/wmt22-comet-da
done

echo "=== Complete. Results at ${EVAL_DIR} ==="
