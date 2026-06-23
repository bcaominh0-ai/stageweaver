# StageWeaver Server Setup

## Project Path

- Project root: `/data/xiezhen/stageweaver`

## Python Environments

- `.venv-cu126`: main StageWeaver validation and A800 CUDA environment.
  - Uses `torch==2.7.0+cu126`.
  - Validate all eight A800 devices and a small CUDA matmul with `scripts/linux_preflight.sh`.

Activate:

```bash
cd /data/xiezhen/stageweaver
source .venv-cu126/bin/activate
```

## SearXNG

SearXNG is deployed from:

```bash
/data/xiezhen/stageweaver/infra/searxng/docker-compose.yml
```

Current deployment uses Docker `host` network and binds only to
`127.0.0.1:18080`. Port 8080 is occupied by another SearXNG deployment on
this server. No outbound proxy is configured for StageWeaver SearXNG.

Check SearXNG:

```bash
docker ps --filter name=stageweaver-searxng
curl -sS --max-time 15 "http://127.0.0.1:18080/search?q=openai&format=json" | head -c 500
```

## Environment Template

Copy `.env.template` or `.env.example` to `.env` and fill local values. Do not commit `.env`.

Common variables to fill:

```bash
OPENAI_BASE_URL=
OPENAI_API_KEY=
JUDGE_BASE_URL=
JUDGE_API_KEY=
JUDGE_MODEL=
META_MODEL=
EXEC_MODEL=
META_MODEL_PATH=
EXEC_MODEL_PATH=
CRAWL_EXTRACT_BASE_URL=
CRAWL_EXTRACT_API_KEY=
CRAWL_EXTRACT_MODEL=
AUDIO_TRANSCRIPTION_BASE_URL=
AUDIO_TRANSCRIPTION_API_KEY=
AUDIO_TRANSCRIPTION_MODEL=qwen3-omni-flash-all
SOMARK_BASE_URL=https://www.dmxapi.cn/v1/responses
SOMARK_API_KEY=
SOMARK_MODEL=somark
SEARXNG_HOST=http://127.0.0.1:18080
```

For `server/documents_tool.py` audio transcription, configure an OpenAI-compatible
endpoint with the `AUDIO_TRANSCRIPTION_*` variables above. A recommended model
value is `qwen3-omni-flash-all`.

For `server/documents_tool.py`, fill `SOMARK_API_KEY` if you want PDFs and
supported image/pdf document types to use Somark first. PDF files fall back to
local `PyPDF2` text extraction if Somark fails. The active Somark endpoint is
configured through `SOMARK_BASE_URL`, with `SOMARK_MODEL` defaulting to `somark`.

## Minimal Runner Smoke

Do not run this until `.env` has been filled with valid endpoint/model settings.

```bash
cd /data/xiezhen/stageweaver
source .venv-cu126/bin/activate
python client/stageweaver_runner.py \
  --memory_mode memento_text \
  --data_jsonl data/deepresearcher_protocol/seen_dev.jsonl \
  --memory_jsonl result/stageweaver/current/stage_bank/stage_bank_train.jsonl \
  --limit 1 \
  --judge_mode exact_match \
  --output_dir /tmp/stageweaver_runner_smoke
```
