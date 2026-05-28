#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export REPO_ROOT
export SEARXNG_HOST="${SEARXNG_HOST:-http://127.0.0.1:8080}"

fail() {
  echo "[linux_preflight] $1" >&2
  exit 1
}

check_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "Required file is missing: $path"
}

cd "$REPO_ROOT"

echo "[linux_preflight] Python version"
"$PYTHON_BIN" --version

echo "[linux_preflight] Torch / CUDA / GPU info"
"$PYTHON_BIN" - <<'PY'
import torch
print({
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
})
PY

echo "[linux_preflight] Core imports"
"$PYTHON_BIN" - <<'PY'
import accelerate
import modelscope
import sentence_transformers
import transformers
print({
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "modelscope": modelscope.__version__,
})
PY

echo "[linux_preflight] Protocol split files"
check_file "$REPO_ROOT/data/deepresearcher_protocol/seen_train.jsonl"
check_file "$REPO_ROOT/data/deepresearcher_protocol/seen_dev.jsonl"
check_file "$REPO_ROOT/data/deepresearcher_protocol/ood_test.jsonl"

echo "[linux_preflight] SearXNG backend"
if ! command -v curl >/dev/null 2>&1; then
  fail "curl is required to check SearXNG"
fi
status_code="$(curl -sS -o /tmp/stageweaver_searxng_preflight.json -w '%{http_code}' "${SEARXNG_HOST%/}/search?q=stageweaver&format=json" || true)"
if [[ "$status_code" != "200" ]]; then
  fail "SearXNG search endpoint is not reachable at $SEARXNG_HOST (HTTP $status_code)"
fi
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/stageweaver_searxng_preflight.json").read_text(encoding="utf-8"))
if not isinstance(payload.get("results", []), list):
    raise SystemExit("SearXNG JSON response does not contain a results list")
print({"query": payload.get("query"), "results": len(payload.get("results", []))})
PY

echo "[linux_preflight] No active archive imports"
if command -v rg >/dev/null 2>&1; then
  if rg -n --glob '!scripts/linux_preflight.sh' "archive\.|from archive|import archive" client memory server scripts; then
    fail "Active code imports archive/"
  fi
else
  if find client memory server scripts -type f ! -path "scripts/linux_preflight.sh" -print0 | xargs -0 grep -InE "archive\.|from archive|import archive"; then
    fail "Active code imports archive/"
  fi
fi

echo "[linux_preflight] Semantic stage names"
required_stages=(
  collect_seen_train_traces
  collect_seen_dev_traces
  build_stageweaver_bank
  train_append_sft
  eval_seen_dev_gate
  eval_ood_test
)
for stage_name in "${required_stages[@]}"; do
  if ! grep -q "$stage_name" "$REPO_ROOT/docs/CURRENT_PROTOCOL.md"; then
    fail "Missing semantic stage name in docs/CURRENT_PROTOCOL.md: $stage_name"
  fi
done

echo "[linux_preflight] Semantic workflow scripts"
for script_name in \
  collect_seen_train_traces.sh \
  collect_seen_dev_traces.sh \
  build_stageweaver_bank.sh \
  train_append_sft.sh \
  eval_seen_dev_gate.sh \
  eval_ood_test.sh; do
  check_file "$REPO_ROOT/scripts/$script_name"
done

echo "[linux_preflight] OK"
