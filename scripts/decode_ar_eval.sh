#!/bin/bash
# AR baseline WMT22 decode + BLEU/COMET-22 eval driver (release version).
#
# Usage: bash scripts/decode_ar_eval.sh <run_id> <lang_pair> <seed> <run_dir>
#   run_id    arbitrary self-describing tag (e.g. ar_enzh_seed42)
#   lang_pair en-zh | zh-en | en-de
#   seed      42 | 123 | 456
#   run_dir   subdir under $OUTPUT_ROOT, e.g. ar_enzh_seed42
#
# Environment:
#   PROJECT_DIR  repo root (default: $PWD)
#   BASE_MODEL   path to Meta-Llama-3-8B-Base local checkpoint
#   OUTPUT_ROOT  where LoRA checkpoints live (default: $PROJECT_DIR/checkpoints/ar_baseline)
#   TEST_DIR     directory containing wmt22_{enzh,ende}_test.jsonl (Zh→En reuses wmt22_enzh_test.jsonl with src/tgt keys swapped; default: $PROJECT_DIR/data)
#   MAX_EXAMPLES default -1 = all (2037 for WMT22 En<->Zh / En->De test)
#   GPU          CUDA_VISIBLE_DEVICES (default: 0)
#   CONDA_ENV    conda env (default: ladit)

set -euo pipefail

RUN_ID="${1:?run_id required}"
LANG_PAIR="${2:?lang_pair required}"
SEED="${3:?seed required}"
RUN_DIR="${4:?run_dir required}"

MAX_EXAMPLES="${MAX_EXAMPLES:--1}"
GPU="${GPU:-0}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
BASE_MODEL="${BASE_MODEL:-/path/to/Meta-Llama-3-8B-Base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/checkpoints/ar_baseline}"
TEST_DIR="${TEST_DIR:-${PROJECT_DIR}/data}"
CONDA_ENV="${CONDA_ENV:-ladit}"

LORA_PATH="${OUTPUT_ROOT}/${RUN_DIR}/final"

case "$LANG_PAIR" in
  en-zh) TEST_FILE="${TEST_DIR}/wmt22_enzh_test.jsonl"; SRC_KEY="en"; REF_KEY="zh"; TGT_LANG="zh" ;;
  zh-en) TEST_FILE="${TEST_DIR}/wmt22_enzh_test.jsonl"; SRC_KEY="zh"; REF_KEY="en"; TGT_LANG="en" ;;
  en-de) TEST_FILE="${TEST_DIR}/wmt22_ende_test.jsonl"; SRC_KEY="en"; REF_KEY="de"; TGT_LANG="de" ;;
  *) echo "ERROR: unsupported lang_pair $LANG_PAIR"; exit 1 ;;
esac

OUT_DIR="${PROJECT_DIR}/eval_results/${RUN_ID}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_ID}_ar_eval.log"

# Split JSONL test file into parallel src/ref txt files (one sentence per line)
SRC_FILE="${OUT_DIR}/src.txt"
REF_FILE="${OUT_DIR}/ref.txt"
python -c "
import json, sys
with open('${TEST_FILE}') as f, open('${SRC_FILE}', 'w') as fs, open('${REF_FILE}', 'w') as fr:
    for line in f:
        ex = json.loads(line)
        fs.write(ex['${SRC_KEY}'].replace('\n', ' ') + '\n')
        fr.write(ex['${REF_KEY}'].replace('\n', ' ') + '\n')
"

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$PROJECT_DIR"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

echo "[$(date)] start AR eval: $RUN_ID lang=$LANG_PAIR seed=$SEED dir=$RUN_DIR" | tee "$LOG_FILE"

# 1) Decode (beam=4, max_new_tokens=256)
python -u -m ladit.evaluation.decode_ar \
    --base_model_path "$BASE_MODEL" \
    --lora_path       "$LORA_PATH" \
    --src_file        "$SRC_FILE" \
    --ref_file        "$REF_FILE" \
    --output_dir      "$OUT_DIR" \
    --lang_pair       "$LANG_PAIR" \
    --num_beams       4 \
    --max_new_tokens  256 \
    --batch_size      4 \
    --max_examples    "$MAX_EXAMPLES" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] decode done, running evaluate.py" | tee -a "$LOG_FILE"

# 2) Eval (BLEU + COMET-22)
python -u -m ladit.evaluation.evaluate \
    --translations_file "$OUT_DIR/translations.json" \
    --output_file       "$OUT_DIR/eval_metrics.json" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] AR_EVAL_COMPLETE $RUN_ID" | tee -a "$LOG_FILE"
