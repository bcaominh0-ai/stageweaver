#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-cu126/bin/python}"
OUTPUT_DIR="${REPO_ROOT}/result/stageweaver/current/tracebank_seen_dev"
TRACE_JSONL="$OUTPUT_DIR/trace_collect_seen_dev.jsonl"
export SEARXNG_HOST="${SEARXNG_HOST:-http://127.0.0.1:18080}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" client/stageweaver_runner.py \
  --memory_mode none \
  --diagnostic_trace_bank \
  --judge_mode exact_match \
  --data_jsonl data/deepresearcher_protocol/seen_dev.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --trace_jsonl "$TRACE_JSONL" \
  --limit 0 \
  "$@"
