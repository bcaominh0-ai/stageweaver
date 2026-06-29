# StageWeaver

StageWeaver 是当前主线实验仓库，用于 DeepResearcher 协议下的记忆构建、Memento-Text 对照和 StageWeaver 潜变量记忆评测。

当前只保留两个有效对比模式：

- `memento_text`：从当前 StageWeaver 记忆库中检索 planner 文本案例，作为文本记忆条件。
- `stageweaver`：按 `agent_role` 和 `stage` 检索 StageWeaver 角色记忆，并转换为 latent memory。

`none` 只用于诊断和采集 trace bank，必须配合 `--diagnostic_trace_bank` 使用。

## 环境

当前 A800 服务器使用 CUDA 12.6 环境：

```bash
cd /data/xiezhen/stageweaver
conda activate /data/xiezhen/stageweaver/.venv-cu126
```

脚本默认会优先使用：

```bash
.venv-cu126/bin/python
```

如果需要临时覆盖，可以显式传入：

```bash
PYTHON_BIN=/path/to/python bash scripts/linux_preflight.sh
```

当前环境面向 8 卡 NVIDIA A800-SXM4-80GB：

- Python `3.11`
- torch `2.7.0+cu126`
- torchvision `0.22.0+cu126`
- torchaudio `2.7.0+cu126`

依赖复现入口：

- `requirements-linux-a800-cu126.txt`
- `pyproject.toml`
- `uv.lock`
- `environment-linux-a800.yml`

## 配置

复制 `.env.template` 为 `.env`，并填写本地端点：

```bash
cp .env.template .env
```

关键变量：

- `TEACHER_BASE_URL` / `TEACHER_API_KEY` / `TEACHER_MODEL`：采集训练轨迹时使用的教师模型。
- `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL`：LLM judge 使用的模型。
- `OPENAI_BASE_URL` / `OPENAI_API_KEY`：普通 OpenAI-compatible agent 后端。
- `SEARXNG_HOST=http://119.45.1.210` / `SEARXNG_API_KEY`：公网搜索 API 地址和鉴权 key，以 `.env` 配置为准。
- `STAGEWEAVER_BANK_JSONL`：默认记忆库路径。
- `STAGEWEAVER_COMPOSER_CKPT`：StageWeaver composer checkpoint。

采集 trace bank 时，`memory_mode=none` 加 `--diagnostic_trace_bank` 会默认读取 `TEACHER_*` 作为 planner/executor 模型端点。命令行显式传入的 `--openai_base_url`、`--openai_api_key`、`--meta_model`、`--exec_model` 仍然优先。

## SearXNG

StageWeaver 的搜索工具通过 `.env` 中配置的公网搜索 API 调用 SearXNG；当前项目环境使用：

```bash
SEARXNG_HOST=http://119.45.1.210
SEARXNG_URL=http://119.45.1.210
SEARXNG_API_STYLE=api
SEARXNG_API_KEY=你的真实API_KEY
SEARXNG_API_KEY_PLACEMENT=query
SEARXNG_API_KEY_PARAM=api_key
SEARXNG_LANGUAGE_PARAM=lang
SEARXNG_ENGINES=bing
```

检查服务连通性：

```bash
curl -i --connect-timeout 5 --max-time 10 "${SEARXNG_HOST%/}/health"
```

检查搜索调用：

```bash
curl -sS -G --connect-timeout 5 --max-time 30 "${SEARXNG_HOST%/}/search" \
  --data-urlencode "q=stageweaver" \
  --data-urlencode "engines=${SEARXNG_ENGINES:-bing}" \
  --data-urlencode "limit=5" \
  --data-urlencode "lang=en" \
  --data-urlencode "api_key=$SEARXNG_API_KEY" | head -c 500
```

如需临时在本机部署 SearXNG，可使用：

```bash
docker compose -f infra/searxng/docker-compose.yml up -d
```

本机部署配置文件在：

- `infra/searxng/docker-compose.yml`
- `infra/searxng/settings.yml`


## Remote Crawl API

StageWeaver 的 `crawl_extract` 现在优先调用公网远程 Crawl API，而不是只依赖本机直接打开网页或 search API 摘要。这样可以把公网访问、网页抓取、反爬重试、搜索兜底放到云服务器上执行，实验服务器只需要通过 HTTP API 获取可抽取的页面 Markdown。

当前远程服务地址：

```bash
http://119.45.1.210/crawl_extract
```

StageWeaver 侧 `.env` 需要配置：

```bash
REMOTE_CRAWL_EXTRACT_URL=http://119.45.1.210/crawl_extract
REMOTE_CRAWL_EXTRACT_API_KEY=你的真实_API_KEY
REMOTE_CRAWL_EXTRACT_TIMEOUT_SEC=60
REMOTE_CRAWL_EXTRACT_BYPASS_PROXY=1
```

说明：

- `REMOTE_CRAWL_EXTRACT_URL`：远程 crawl API 地址。
- `REMOTE_CRAWL_EXTRACT_API_KEY`：远程 API Key。可以和搜索 API 的 key 保持一致，但不要提交真实 key。
- `REMOTE_CRAWL_EXTRACT_TIMEOUT_SEC`：StageWeaver 等待远程 crawl API 的超时时间。
- `REMOTE_CRAWL_EXTRACT_BYPASS_PROXY=1`：调用远程公网 API 时绕过实验服务器本地代理设置，避免错误代理导致连接失败。

运行流程：

1. executor 调用 `crawl_extract(url, query)`。
2. `server/ai_crawl.py` 检查 `REMOTE_CRAWL_EXTRACT_URL` 是否配置。
3. 如果已配置，StageWeaver 向远程 `/crawl_extract` 发送 `url`、`query`、`max_chars` 和 API Key。
4. 远程 API 优先直接打开目标网页，抽取 title、meta、标题、段落、列表等正文内容，并返回相关 Markdown。
5. 如果远程直接抓取失败或内容为空，远程 API 会使用云服务器上的 SearXNG 搜索摘要作为 fallback。
6. 如果远程 API 调用失败，StageWeaver 仍可按 `CRAWL_SEARCH_FALLBACK` 配置退回本地 search fallback。
7. StageWeaver 再使用 `CRAWL_EXTRACT_MODEL` 从远程返回的 Markdown 中抽取最终答案。

远程 API 返回的关键字段：

```json
{
  "status": "ok",
  "source": "direct",
  "url": "https://example.com",
  "final_url": "https://example.com",
  "status_code": 200,
  "content_type": "text/html; charset=UTF-8",
  "title": "Example Domain",
  "query": "example query",
  "markdown": "...",
  "chars": 1234,
  "chunks_found": 10,
  "chunks_returned": 8,
  "elapsed_sec": 1.23
}
```

`source` 的含义：

- `direct`：远程服务成功直接打开目标 URL，并从页面 HTML 中抽取内容。
- `search_fallback`：直接抓取失败，远程服务退回搜索摘要。

常用测试：

```bash
curl -s http://119.45.1.210/health

curl -sG "http://119.45.1.210/crawl_extract" \
  --data-urlencode "url=https://en.wikipedia.org/wiki/OpenAI" \
  --data-urlencode "query=When was OpenAI founded and who founded it?" \
  --data-urlencode "api_key=你的真实_API_KEY"
```

StageWeaver 内部测试：

```bash
cd /data/xiezhen/stageweaver
/data/xiezhen/stageweaver/.venv-cu126/bin/python - <<'PY'
import asyncio
from server.ai_crawl import crawl_extract

async def main():
    result = await crawl_extract(
        "https://en.wikipedia.org/wiki/OpenAI",
        "When was OpenAI founded and who founded it?",
        max_tokens=500,
    )
    print(result[:1000])

asyncio.run(main())
PY
```

远程服务部署在云服务器 `/home/ubuntu/search_api`，由 Docker Compose 的 `api` 服务对外暴露 80 端口，并复用同一套 API Key 鉴权。服务内置 SSRF 防护，只允许 `http` / `https`，并拒绝 localhost、内网地址和重定向到内网地址的请求。

### 云端 API 加速配置

2026-06-28 对 `http://119.45.1.210` 上的 search/crawl API 做了第一阶段加速升级，目标是不改变 StageWeaver 调用协议，只提升并发能力、减少重复请求耗时，并增加可观测字段。

云端实际改动：

- `api` 服务使用 granian 多 worker：`GRANIAN_WORKERS=3`。
- 每个 worker 使用 `GRANIAN_BLOCKING_THREADS=8`，适配 4 核 CPU，避免线程数过高。
- `/search` 和 `/crawl_extract` 复用全局 `httpx.Client` 连接池：
  - `HTTP_MAX_CONNECTIONS=64`
  - `HTTP_MAX_KEEPALIVE_CONNECTIONS=16`
- 增加 Redis TTL cache：
  - `SEARCH_CACHE_TTL_SECONDS=1800`
  - `CRAWL_CACHE_TTL_SECONDS=21600`
- `/search` 和 `/crawl_extract` 响应新增兼容字段：
  - `cache_hit`：是否命中 Redis 缓存。
  - `elapsed_sec`：API 服务端处理耗时。
- `/health` 会返回 cache 和 HTTP 连接池配置，便于远程检查。
- API 日志会记录 search/crawl 的 `cache_hit`、`source`、`count`、`chars` 和 `elapsed`。

云端同步文件：

```text
/home/ubuntu/search_api/api/app/main.py
/home/ubuntu/search_api/docker-compose.yml
```

升级后验证结果：

```text
/health: cache enabled, search TTL 1800s, crawl TTL 21600s
search miss: 约 3.2s
search hit: 约 0.002s
crawl_extract miss: 约 1.2s，source=direct
crawl_extract hit: 约 0.002s
```

后续如果迁移到新云服务器，需要同步上述两个文件，并确保 compose 中存在 `redis` 服务和以下环境变量：

```bash
CACHE_ENABLED=1
REDIS_HOST=redis
REDIS_PORT=6379
SEARCH_CACHE_TTL_SECONDS=1800
CRAWL_CACHE_TTL_SECONDS=21600
HTTP_MAX_CONNECTIONS=64
HTTP_MAX_KEEPALIVE_CONNECTIONS=16
GRANIAN_WORKERS=3
GRANIAN_BLOCKING_THREADS=8
```

## 数据协议

当前使用 DeepResearcher 三个固定 split：

- `data/deepresearcher_protocol/seen_train.jsonl`
- `data/deepresearcher_protocol/seen_dev.jsonl`
- `data/deepresearcher_protocol/ood_test.jsonl`

不要重新 shuffle 或生成 split。正式汇报 `ood_test` 之前，先用 `seen_dev` 做 gate。


## 已整理 traces

为了方便直接阅读教师轨迹，当前 train/dev traces 已集中放在：

```text
result/stageweaver/current/readable_traces/
```

目录内容：

- `train_traces.jsonl`：来自 `seen_train` 的教师 trace，共 1188 条。
- `dev_traces/`：合并后的 `seen_dev` tracebank，使用旧 dev `index 0-299` 和新 Qwen3-Next-80B `index 300-411`；重复的 `index=300` 以新目录为准。

`dev_traces/` 内主要文件：

- `trace_collect_seen_dev_merged.jsonl`：合并后的 dev 教师 traces，共 361 条。
- `results_none_full_both_merged.jsonl`：合并后的 dev 运行结果，共 412 条，覆盖 `index 0-411`。
- `merge_manifest.json`：合并来源、规则、缺失 trace index 与统计信息。

## 当前工作流

推荐按下面顺序执行：

```bash
bash scripts/linux_preflight.sh
bash scripts/collect_seen_train_traces.sh
bash scripts/collect_seen_dev_traces.sh
bash scripts/build_stageweaver_bank.sh
bash scripts/train_append_sft.sh
bash scripts/eval_seen_dev_gate.sh
bash scripts/eval_ood_test.sh
```

各阶段输出目录：

- `result/stageweaver/current/readable_traces/train_traces.jsonl`
- `result/stageweaver/current/readable_traces/dev_traces/`
- `result/stageweaver/current/stage_bank/`
- `result/stageweaver/current/append_sft/`
- `result/stageweaver/current/eval_seen_dev/`
- `result/stageweaver/current/eval_ood_test/`

## 构建记忆库

先采集 seen train/dev 轨迹：

```bash
bash scripts/collect_seen_train_traces.sh
bash scripts/collect_seen_dev_traces.sh
```

再构建 StageWeaver 记忆库：

```bash
bash scripts/build_stageweaver_bank.sh
```

默认输出：

```text
result/stageweaver/current/stage_bank/stage_bank_train.jsonl
result/stageweaver/current/stage_bank/stage_bank_val.jsonl
result/stageweaver/current/stage_bank/stage_bank_test.jsonl
result/stageweaver/current/stage_bank/stage_bank_stats.json
```

成功轨迹会生成 `success_case`。失败轨迹如果需要蒸馏 insight，可以在 `build_stageweaver_bank.py` 中使用 `--distill_failure_insights`，并配置 `INSIGHT_*`。

## Composer SFT

当前训练结构与 LatentMem 对齐：检索到的 StageTuple 文本先输入 Qwen3-4B Composer；Composer 使用 `q_proj`/`v_proj` LoRA（默认 `r=16`、`alpha=32`、`dropout=0.1`）和 8 个可训练 query latents。得到的 latent hidden states 经单层 `Linear(2560, 2560)` 投影后，追加到冻结的 Agent Qwen3-4B 输入 embedding 中，并以回答 token 的交叉熵联合训练 LoRA、query latents 和 projector。

```bash
QWEN3_4B_MODEL_PATH=/data/xiezhen/llm/models/Qwen3-4B-Instruct-2507 \
  bash scripts/train_append_sft.sh
```

脚本在每次训练前都会读取 GPU 显存和利用率，只允许使用显存占用不超过 100 MiB 且利用率不超过 5% 的单张卡。显式设置 `CUDA_VISIBLE_DEVICES` 时同样会校验该物理卡，卡忙、编号不存在或指定多卡都会拒绝启动；未指定时才自动选择空闲卡。当前入口实现的是 CE SFT，LMPO reward 训练属于后续阶段，尚未接入此脚本。

## 评测

Memento-Text 默认读取：

```text
result/stageweaver/current/stage_bank/stage_bank_train.jsonl
```

StageWeaver 在线评测需要：

- local direct backend
- append-aligned composer checkpoint
- `--stage_mode both`
- `--latent_interface append`
- `--agent_backend local`

先跑 seen-dev gate：

```bash
bash scripts/eval_seen_dev_gate.sh
```

再跑 OOD test：

```bash
bash scripts/eval_ood_test.sh
```

## 工具

当前默认加载六类工具脚本：

- `server/code_agent.py`
- `server/documents_tool.py`
- `server/image_tool.py`
- `server/math_tool.py`
- `server/ai_crawl.py`
- `server/search_tool.py`

`server/video_tool.py` 不在当前默认工具集内。

## 重要约定

- 当前协议不再使用 legacy `memory/memory.jsonl`。
- 当前协议不再使用 `memento_text_seed_memory.jsonl`。
- workflow 名称使用语义命名，不使用历史 R-run ID。
- 结果目录统一放在 `result/stageweaver/current/` 下。
- `.env` 不要提交。

更多细节见：

- `docs/CURRENT_PROTOCOL.md`
- `docs/CLEANUP_MANIFEST.md`
- `docs/ARCHIVED_RESULTS_SUMMARY.md`
