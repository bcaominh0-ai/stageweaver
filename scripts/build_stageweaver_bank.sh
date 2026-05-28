#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${REPO_ROOT}/result/stageweaver/current/stage_bank"
TRAIN_TRACE_JSONL="${REPO_ROOT}/result/stageweaver/current/tracebank_seen_train/trace_collect_seen_train.jsonl"
TRAIN_RESULTS_JSONL="${REPO_ROOT}/result/stageweaver/current/tracebank_seen_train/results_none_full_both.jsonl"
VAL_TRACE_JSONL="${REPO_ROOT}/result/stageweaver/current/tracebank_seen_dev/trace_collect_seen_dev.jsonl"
VAL_RESULTS_JSONL="${REPO_ROOT}/result/stageweaver/current/tracebank_seen_dev/results_none_full_both.jsonl"

for required_file in "$TRAIN_TRACE_JSONL" "$TRAIN_RESULTS_JSONL" "$VAL_TRACE_JSONL" "$VAL_RESULTS_JSONL"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing prerequisite file: $required_file" >&2
    echo "Run collect_seen_train_traces or collect_seen_dev_traces first." >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" memory/build_stageweaver_bank.py \
  --train_trace_jsonl "$TRAIN_TRACE_JSONL" \
  --train_results_jsonl "$TRAIN_RESULTS_JSONL" \
  --val_trace_jsonl "$VAL_TRACE_JSONL" \
  --val_results_jsonl "$VAL_RESULTS_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
