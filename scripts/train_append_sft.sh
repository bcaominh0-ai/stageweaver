#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${REPO_ROOT}/result/stageweaver/current/append_sft"
TRAIN_JSONL="${REPO_ROOT}/result/stageweaver/current/stage_bank/stage_bank_train.jsonl"
VAL_JSONL="${REPO_ROOT}/result/stageweaver/current/stage_bank/stage_bank_val.jsonl"
RETRIEVAL_JSONL="${REPO_ROOT}/result/stageweaver/current/stage_bank/stage_bank_train.jsonl"
COMPOSER_MODEL_PATH="${COMPOSER_MODEL_PATH:-${QWEN3_4B_MODEL_PATH:-}}"
AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-${QWEN3_4B_MODEL_PATH:-}}"

for required_file in "$TRAIN_JSONL" "$VAL_JSONL"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing prerequisite file: $required_file" >&2
    echo "Run build_stageweaver_bank first." >&2
    exit 1
  fi
done
if [[ -z "$COMPOSER_MODEL_PATH" ]]; then
  echo "Set COMPOSER_MODEL_PATH or QWEN3_4B_MODEL_PATH before running train_append_sft." >&2
  exit 1
fi
if [[ -z "$AGENT_MODEL_PATH" ]]; then
  echo "Set AGENT_MODEL_PATH or QWEN3_4B_MODEL_PATH before running train_append_sft." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" memory/train_stageweaver_composer_sft.py \
  --train_jsonl "$TRAIN_JSONL" \
  --val_jsonl "$VAL_JSONL" \
  --retrieval_jsonl "$RETRIEVAL_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --composer_model_path "$COMPOSER_MODEL_PATH" \
  --agent_model_path "$AGENT_MODEL_PATH" \
  "$@"
