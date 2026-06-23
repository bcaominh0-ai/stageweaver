#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-cu126/bin/python}"
OUTPUT_DIR="${REPO_ROOT}/result/stageweaver/current/append_sft"
TRAIN_JSONL="${REPO_ROOT}/result/stageweaver/current/stage_bank/stage_bank_train.jsonl"
VAL_JSONL="${REPO_ROOT}/result/stageweaver/current/stage_bank/stage_bank_val.jsonl"
RETRIEVAL_JSONL="${REPO_ROOT}/result/stageweaver/current/stage_bank/stage_bank_train.jsonl"
COMPOSER_MODEL_PATH="${COMPOSER_MODEL_PATH:-${QWEN3_4B_MODEL_PATH:-}}"
AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-${QWEN3_4B_MODEL_PATH:-}}"

export HOME="${REPO_ROOT}/.home"
export TMPDIR="${REPO_ROOT}/.cache/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"
export HF_HOME="${REPO_ROOT}/hf_cache"
export MODELSCOPE_CACHE="${REPO_ROOT}/hf_cache/modelscope"
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"

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

mkdir -p \
  "$OUTPUT_DIR" \
  "$HOME" \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$MODELSCOPE_CACHE" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME"
cd "$REPO_ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to verify an idle GPU before every training run." >&2
  exit 1
fi
mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

gpu_is_idle() {
  local requested_index="$1"
  local row index memory_used utilization
  for row in "${gpu_rows[@]}"; do
    IFS=',' read -r index memory_used utilization <<< "$row"
    index="${index//[[:space:]]/}"
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if [[ "$index" == "$requested_index" ]]; then
      (( memory_used <= 100 && utilization <= 5 ))
      return
    fi
  done
  return 1
}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  requested_gpu="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
  if [[ ! "$requested_gpu" =~ ^[0-9]+$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index; got '${CUDA_VISIBLE_DEVICES}'." >&2
    exit 1
  fi
  if ! gpu_is_idle "$requested_gpu"; then
    echo "Requested physical GPU ${requested_gpu} is missing or busy; training was not started." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$requested_gpu"
  echo "Verified caller-selected physical GPU ${requested_gpu} is idle."
else
  selected_gpu=""
  for row in "${gpu_rows[@]}"; do
    IFS=',' read -r index memory_used utilization <<< "$row"
    index="${index//[[:space:]]/}"
    if gpu_is_idle "$index"; then
      selected_gpu="$index"
      break
    fi
  done
  if [[ -z "$selected_gpu" ]]; then
    echo "No idle GPU found (requires <=100 MiB memory and <=5% utilization); training was not started." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$selected_gpu"
  echo "Selected idle physical GPU ${selected_gpu}."
fi

"$PYTHON_BIN" memory/train_stageweaver_composer_sft.py \
  --train_jsonl "$TRAIN_JSONL" \
  --val_jsonl "$VAL_JSONL" \
  --retrieval_jsonl "$RETRIEVAL_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --composer_model_path "$COMPOSER_MODEL_PATH" \
  --agent_model_path "$AGENT_MODEL_PATH" \
  "$@"
