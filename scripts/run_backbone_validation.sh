#!/bin/bash
# Cross-backbone validation pipeline for the paper's "Cross-Backbone Validation"
# appendix. Reproduces a single (backbone, seed, language-pair) cell of that
# table by running LoRA-SFT, decoding under the three length policies
# (Oracle / fixed-Ratio / Entropy-Valley), and computing BLEU.
#
# Usage:
#   bash scripts/run_backbone_validation.sh <BACKBONE> <SEED> <LANG_PAIR>
# Where:
#   BACKBONE  in {dream, diffullama, llada}
#   SEED      any non-negative integer (paper used 42, 123, 456)
#   LANG_PAIR in {enzh, ende, zhen}
#
# Environment variables (set before invoking):
#   MODELS_DIR      Directory holding pretrained backbones; defaults to
#                   ${HOME}/models. Each backbone is expected at:
#                     ${MODELS_DIR}/LLaDA-8B-Base
#                     ${MODELS_DIR}/Dream-v0-Base-7B
#                     ${MODELS_DIR}/diffullama
#   OUTPUT_BASE     Where to write checkpoints and eval outputs. Defaults to
#                   ${PWD}/runs.
#   WANDB_ENTITY    (optional) WandB entity for run logging.
#
# Notes:
# - The 200k WMT training corpora and the WMT22 test set are not redistributed
#   here (see README "Data" section for assembly instructions). Place them
#   under data/ before running this script.
# - For multi-seed cross-backbone reproduction, run this script once per
#   (backbone, seed, lang_pair) cell.
set -euo pipefail
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false

BACKBONE="${1:?Usage: $0 <BACKBONE> <SEED> <LANG_PAIR>}"
SEED="${2:?Usage: $0 <BACKBONE> <SEED> <LANG_PAIR>}"
LANG_PAIR="${3:?Usage: $0 <BACKBONE> <SEED> <LANG_PAIR>}"

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/runs}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ===== Backbone-specific model path + LoRA tag =====
case "${BACKBONE}" in
  dream)
    MODEL_PATH="${MODELS_DIR}/Dream-v0-Base-7B"
    BACKBONE_TAG="dream"
    ;;
  diffullama)
    MODEL_PATH="${MODELS_DIR}/diffullama"
    BACKBONE_TAG="diffullama"
    ;;
  llada)
    MODEL_PATH="${MODELS_DIR}/LLaDA-8B-Base"
    BACKBONE_TAG="llada"
    ;;
  *)
    echo "ERROR: BACKBONE must be one of {dream, diffullama, llada}"
    exit 1
    ;;
esac

# ===== Per-direction config (matches the paper's main results) =====
case "${LANG_PAIR}" in
  enzh)
    TRAIN_DATA="data/enzh_train.jsonl"
    DEV_DATA="data/enzh_dev.jsonl"
    TEST_DATA="data/wmt22_enzh_test.jsonl"
    RATIO_BASELINE="0.8"
    CANDIDATE_RATIOS="0.7,0.75,0.8,0.85,0.9"
    LANG_ARG="--lang_pair en-zh"
    DECODE_KEYS="--src_key en --tgt_key zh"
    ;;
  ende)
    TRAIN_DATA="data/ende_train.jsonl"
    DEV_DATA="data/ende_dev.jsonl"
    TEST_DATA="data/wmt22_ende_test.jsonl"
    RATIO_BASELINE="1.8"
    CANDIDATE_RATIOS="1.5,1.6,1.7,1.8,1.9"
    LANG_ARG="--lang_pair en-de"
    DECODE_KEYS="--src_key en --tgt_key de"
    ;;
  zhen)
    TRAIN_DATA="data/enzh_train.jsonl"
    DEV_DATA="data/enzh_dev.jsonl"
    TEST_DATA="data/wmt22_enzh_test.jsonl"
    RATIO_BASELINE="1.2"
    CANDIDATE_RATIOS="1.0,1.1,1.2,1.3,1.4"
    LANG_ARG="--lang_pair zh-en"
    DECODE_KEYS="--src_key zh --tgt_key en"
    ;;
  *)
    echo "ERROR: LANG_PAIR must be one of {enzh, ende, zhen}"
    exit 1
    ;;
esac

RUN_NAME="${BACKBONE_TAG}_seed${SEED}_${LANG_PAIR}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
LOG_FILE="${PROJECT_DIR}/logs/${RUN_NAME}_train.log"
mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}"

WANDB_ARGS=""
if [ -n "${WANDB_ENTITY:-}" ]; then
    WANDB_ARGS="--wandb --wandb_project ladit --wandb_entity ${WANDB_ENTITY} --run_name ${RUN_NAME}"
fi

echo "======================================================================"
echo "  Backbone: ${BACKBONE_TAG}   LangPair: ${LANG_PAIR}   Seed: ${SEED}"
echo "  Model:    ${MODEL_PATH}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Start:    $(date)"
echo "======================================================================"

# ===== Phase 1: LoRA SFT =====
CKPT=""
if ls ${OUTPUT_DIR}/checkpoint-* 1>/dev/null 2>&1; then
    CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* | sort -t- -k2 -n | tail -1)
    echo "Checkpoint already exists: ${CKPT}, skipping training."
else
    echo ""
    echo "===== Phase 1: LoRA SFT (backbone=${BACKBONE_TAG}, seed=${SEED}) ====="
    torchrun --nproc_per_node=8 --master_port=29500 \
        -m ladit.training.train \
        --model_path "${MODEL_PATH}" \
        --backbone "${BACKBONE_TAG}" \
        --use_lora \
        --lora_rank 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --gradient_checkpointing \
        --train_data "${TRAIN_DATA}" \
        --dev_data "${DEV_DATA}" \
        --max_seq_len 1024 \
        --noise_schedule uniform \
        ${LANG_ARG} \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 \
        --weight_decay 0.01 \
        --num_epochs 3 \
        --warmup_ratio 0.05 \
        --lr_scheduler cosine \
        --max_grad_norm 1.0 \
        --seed "${SEED}" \
        --bf16 \
        --output_dir "${OUTPUT_DIR}" \
        --save_steps 500 \
        --eval_steps 500 \
        --logging_steps 10 \
        ${WANDB_ARGS} \
        2>&1 | tee "${LOG_FILE}"

    CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    echo "Training complete. Checkpoint: ${CKPT}"
fi

if [ -z "${CKPT}" ]; then
    echo "ERROR: No checkpoint found after training"
    exit 1
fi

# ===== Phase 2: Decode (Oracle / Ratio / EV) =====
echo ""
echo "===== Phase 2: Decode (32-step MED, full WMT22 test set) ====="
EVAL_DIR="${PROJECT_DIR}/eval_results/${RUN_NAME}"
mkdir -p "${EVAL_DIR}"

for method in oracle ratio entropy_valley; do
    case "${method}" in
        oracle)         LENGTH_ARG="--length_method oracle" ;;
        ratio)          LENGTH_ARG="--length_method ratio --length_ratio ${RATIO_BASELINE}" ;;
        entropy_valley) LENGTH_ARG="--length_method entropy_valley --candidate_ratios ${CANDIDATE_RATIOS}" ;;
    esac
    OUT="${EVAL_DIR}/translations_${method}.json"
    if [ -f "${OUT}" ]; then
        echo "Skip ${method}: ${OUT} exists"
        continue
    fi
    echo "--- decoding ${method} ---"
    CUDA_VISIBLE_DEVICES=0 python -m ladit.decoding.translate \
        --model_path "${MODEL_PATH}" \
        --backbone "${BACKBONE_TAG}" \
        --lora_path "${CKPT}" \
        --input_file "${TEST_DATA}" \
        --output_file "${OUT}" \
        --schedule med \
        --num_steps 32 \
        ${LENGTH_ARG} \
        ${DECODE_KEYS} \
        --device cuda 2>&1 | tee -a "${EVAL_DIR}/decode.log"
done

# ===== Phase 3: Inline BLEU (COMET via compute_crossbb_comet.py) =====
echo ""
echo "===== Phase 3: Inline metrics (sacrebleu); see compute_crossbb_comet.py for COMET-22 ====="
python3 - <<PYEOF | tee "${EVAL_DIR}/metrics.txt"
import json, sacrebleu
from pathlib import Path
eval_dir = Path("${EVAL_DIR}")
test_data_file = "${TEST_DATA}"
lp = "${LANG_PAIR}"
key_map = {"enzh": ("en", "zh"), "zhen": ("zh", "en"), "ende": ("en", "de")}
src_key, tgt_key = key_map[lp]
refs = []
with open(test_data_file) as f:
    for line in f:
        refs.append(json.loads(line).get(tgt_key, ""))
for method in ["oracle", "ratio", "entropy_valley"]:
    out = eval_dir / f"translations_{method}.json"
    if not out.exists():
        print(f"[{method}] SKIP — no translations file")
        continue
    data = json.load(open(out))
    hyps = [d.get("hypothesis", d.get("translation", "")) for d in data]
    bleu_tok = "zh" if tgt_key == "zh" else "13a"
    bleu = sacrebleu.corpus_bleu(hyps, [refs[:len(hyps)]], tokenize=bleu_tok)
    print(f"[{method}] N={len(hyps)} BLEU={bleu.score:.2f}")
PYEOF

echo ""
echo "======================================================================"
echo "  ${RUN_NAME} COMPLETE at $(date)"
echo "  Checkpoint: ${CKPT}"
echo "  Eval:       ${EVAL_DIR}"
echo "======================================================================"
