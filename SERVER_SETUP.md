# StageWeaver Server Setup

## Project Path

- Project root: `/home/ubuntu/projects/stageweaver`

## Python Environments

- `.venv-cu128`: main StageWeaver validation and RTX 5090 CUDA environment.
  - Uses `torch==2.7.0+cu128`.
  - Verified with RTX 5090 compute capability `(12, 0)` and a small CUDA matmul.

Activate:

```bash
cd /home/ubuntu/projects/stageweaver
source .venv-cu128/bin/activate
```

## SearXNG

SearXNG is deployed from:

```bash
/home/ubuntu/projects/stageweaver/infra/searxng/docker-compose.yml
```

Current deployment uses Docker `host` network.

Reason: the server proxy is a host-local proxy:

```bash
http_proxy=http://127.0.0.1:7891
https_proxy=http://127.0.0.1:7891
```

With host networking, the SearXNG container can reach the host proxy at `127.0.0.1:7891`. Do not switch SearXNG to Docker bridge networking unless container access to the host proxy has been separately confirmed. In Docker bridge mode, `127.0.0.1` inside the container is the container itself, not the host, so `http://127.0.0.1:7891` would not reach the host proxy.

Check SearXNG:

```bash
docker ps --filter name=stageweaver-searxng
curl -sS --max-time 10 "http://127.0.0.1:8080/search?q=openai&format=json" | head -c 500
```

## Temporary 8080 Protection

SearXNG currently listens on `*:8080` because it uses host networking. Temporary `iptables` and `ip6tables` rules block non-localhost access to port 8080.

These rules are not persistent. After a server reboot or firewall reset, re-add them or close port 8080 in the cloud security group.

Apply temporary protection:

```bash
sudo iptables -I INPUT -p tcp --dport 8080 ! -s 127.0.0.1 -j DROP
sudo ip6tables -I INPUT -p tcp --dport 8080 ! -s ::1 -j DROP
```

Verify:

```bash
curl -sS --max-time 10 "http://127.0.0.1:8080/search?q=openai&format=json" | head -c 500
sudo iptables -S | grep 8080 || true
sudo ip6tables -S | grep 8080 || true
```

Expected rule summary:

```text
-A INPUT ! -s 127.0.0.1/32 -p tcp -m tcp --dport 8080 -j DROP
-A INPUT ! -s ::1/128 -p tcp -m tcp --dport 8080 -j DROP
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
SEARXNG_HOST=http://127.0.0.1:8080
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
cd /home/ubuntu/projects/stageweaver
source .venv-cu128/bin/activate
python client/stageweaver_runner.py \
  --memory_mode memento_text \
  --data_jsonl data/deepresearcher_protocol/seen_dev.jsonl \
  --memory_jsonl result/stageweaver/current/stage_bank/stage_bank_train.jsonl \
  --limit 1 \
  --judge_mode exact_match \
  --output_dir /tmp/stageweaver_runner_smoke
```
