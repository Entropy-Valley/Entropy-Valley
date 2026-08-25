#!/bin/bash
# AR matched-protocol baseline launcher (release version).
# Trains LLaMA-3-8B-Base + LoRA on the same parallel data as the LLaDA SFT, with
# identical effective batch=128, 3 epochs, and seed schedule. Only the backbone
# and the loss (next-token vs masked-diffusion CE) differ.
#
# Usage: bash scripts/run_ar_baseline.sh <lang_pair> <seed> <run_id>
#   lang_pair: en-zh | zh-en | en-de
#   seed:      42 | 123 | 456 (paper uses 3 seeds)
#   run_id:    arbitrary self-describing tag, e.g. ar_enzh_seed42
#
# Environment (set before invoking, or edit defaults below):
#   PROJECT_DIR  repo root (default: $PWD)
#   MODEL_PATH   path to Meta-Llama-3-8B-Base local checkpoint
#   OUTPUT_ROOT  where to write LoRA checkpoints
#   CONDA_ENV    conda env with torch/transformers/peft (default: ladit)
#   WANDB_*      optional WandB logging
set -euo pipefail

LANG_PAIR="${1:-en-zh}"
SEED="${2:-42}"
RUN_ID="${3:-ar_run}"

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MODEL_PATH="${MODEL_PATH:-/path/to/Meta-Llama-3-8B-Base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/checkpoints/ar_baseline}"
CONDA_ENV="${CONDA_ENV:-ladit}"

case "$LANG_PAIR" in
  en-zh) TRAIN="data/enzh_train.jsonl"; DEV="data/enzh_dev.jsonl"; SLUG="enzh" ;;
  zh-en) TRAIN="data/enzh_train.jsonl"; DEV="data/enzh_dev.jsonl"; SLUG="zhen" ;;
  en-de) TRAIN="data/ende_train.jsonl"; DEV="data/ende_dev.jsonl"; SLUG="ende" ;;
  *) echo "ERROR: bad lang_pair $LANG_PAIR"; exit 1 ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}_ar_${SLUG}_seed${SEED}"
RUN_NAME="${RUN_ID}_ar_${SLUG}_seed${SEED}"
LOG_FILE="${PROJECT_DIR}/logs/${RUN_NAME}_train.log"

mkdir -p "$(dirname "$LOG_FILE")" "$OUTPUT_DIR"

# Hard rules: unbuffered stdout so val metrics survive process exit
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# 8 GPUs, per-device micro=4, grad_accum=4 => global batch = 128 (matches LLaDA SFT)
torchrun --standalone --nproc_per_node=8 -m ladit.training.train_ar \
    --model_path        "$MODEL_PATH" \
    --train_data        "$TRAIN" \
    --dev_data          "$DEV" \
    --max_seq_len       1024 \
    --lang_pair         "$LANG_PAIR" \
    --batch_size        4 \
    --gradient_accumulation_steps 4 \
    --learning_rate     2e-4 \
    --weight_decay      0.01 \
    --num_epochs        3 \
    --warmup_ratio      0.05 \
    --lr_scheduler      cosine \
    --max_grad_norm     1.0 \
    --gradient_checkpointing \
    --bf16 \
    --use_lora \
    --lora_rank         64 \
    --lora_alpha        128 \
    --lora_dropout      0.05 \
    --seed              "$SEED" \
    --output_dir        "$OUTPUT_DIR" \
    --save_steps        500 \
    --save_total_limit  2 \
    --logging_steps     10 \
    --run_name          "$RUN_NAME" \
    2>&1 | tee "$LOG_FILE"
