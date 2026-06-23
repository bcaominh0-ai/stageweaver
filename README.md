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
source .venv-cu126/bin/activate
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
- `SEARXNG_HOST=http://127.0.0.1:18080`：搜索后端。
- `STAGEWEAVER_BANK_JSONL`：默认记忆库路径。
- `STAGEWEAVER_COMPOSER_CKPT`：StageWeaver composer checkpoint。

采集 trace bank 时，`memory_mode=none` 加 `--diagnostic_trace_bank` 会默认读取 `TEACHER_*` 作为 planner/executor 模型端点。命令行显式传入的 `--openai_base_url`、`--openai_api_key`、`--meta_model`、`--exec_model` 仍然优先。

## SearXNG

StageWeaver 的搜索工具依赖本地 SearXNG。

启动：

```bash
docker compose -f infra/searxng/docker-compose.yml up -d
```

检查：

```bash
curl -sS "http://127.0.0.1:18080/search?q=stageweaver&format=json" | head -c 500
```

当前 SearXNG 使用 Docker host network，配置文件在：

- `infra/searxng/docker-compose.yml`
- `infra/searxng/settings.yml`

## 数据协议

当前使用 DeepResearcher 三个固定 split：

- `data/deepresearcher_protocol/seen_train.jsonl`
- `data/deepresearcher_protocol/seen_dev.jsonl`
- `data/deepresearcher_protocol/ood_test.jsonl`

不要重新 shuffle 或生成 split。正式汇报 `ood_test` 之前，先用 `seen_dev` 做 gate。

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

- `result/stageweaver/current/tracebank_seen_train/`
- `result/stageweaver/current/tracebank_seen_dev/`
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
