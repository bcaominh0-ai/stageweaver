#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-cu128/bin/python}"
OUTPUT_DIR="${REPO_ROOT}/result/stageweaver/current/eval_seen_dev"
DATA_JSONL="${REPO_ROOT}/data/deepresearcher_protocol/seen_dev.jsonl"
MEMORY_MODE="${MEMORY_MODE:-stageweaver}"
JUDGE_MODE="${JUDGE_MODE:-llm}"
META_MODEL_PATH="${META_MODEL_PATH:-${QWEN3_4B_MODEL_PATH:-}}"
EXEC_MODEL_PATH="${EXEC_MODEL_PATH:-${QWEN3_4B_MODEL_PATH:-}}"
STAGEWEAVER_COMPOSER_CKPT="${STAGEWEAVER_COMPOSER_CKPT:-${REPO_ROOT}/result/stageweaver/current/append_sft/stageweaver_composer_sft.pt}"
export SEARXNG_HOST="${SEARXNG_HOST:-http://127.0.0.1:8080}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

COMMON_ARGS=(
  --memory_mode "$MEMORY_MODE"
  --judge_mode "$JUDGE_MODE"
  --data_jsonl "$DATA_JSONL"
  --output_dir "$OUTPUT_DIR"
  --limit 0
)

if [[ "$MEMORY_MODE" == "stageweaver" ]]; then
  if [[ ! -f "$STAGEWEAVER_COMPOSER_CKPT" ]]; then
    echo "Missing append-SFT checkpoint: $STAGEWEAVER_COMPOSER_CKPT" >&2
    echo "Run train_append_sft first." >&2
    exit 1
  fi
  if [[ -z "$META_MODEL_PATH" || -z "$EXEC_MODEL_PATH" ]]; then
    echo "Set META_MODEL_PATH and EXEC_MODEL_PATH (or QWEN3_4B_MODEL_PATH) before StageWeaver evaluation." >&2
    exit 1
  fi
  COMMON_ARGS+=(
    --agent_backend local
    --meta_model_path "$META_MODEL_PATH"
    --exec_model_path "$EXEC_MODEL_PATH"
    --stageweaver_composer_ckpt "$STAGEWEAVER_COMPOSER_CKPT"
  )
fi

"$PYTHON_BIN" client/stageweaver_runner.py "${COMMON_ARGS[@]}" "$@"
