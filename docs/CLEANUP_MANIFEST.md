# Cleanup Manifest

## Active mainline files to keep

- `client/stageweaver_runner.py`: canonical runner for `memento_text` and `stageweaver`.
- `client/agent_local_server.py`: planner/executor runtime with append latent support and MCP tool wiring.
- `memory/stageweaver_schema.py`: StageWeaver tuple schema and role-conditioned retrieval text.
- `memory/build_stageweaver_bank.py`: converts current traces into StageWeaver stage tuples.
- `memory/stageweaver_serializers.py`: serializes retrieved positive outputs for latent composition.
- `memory/stageweaver_composer.py`: LatentMem-style composer.
- `memory/stageweaver_projector.py`: composer-to-agent hidden-size projector.
- `memory/stageweaver_dataset.py`: current composer SFT dataset construction.
- `memory/stageweaver_losses.py`: lightweight StageWeaver loss helper.
- `memory/train_stageweaver_composer_sft.py`: append-aligned composer SFT entrypoint.
- `memory/semantic_retriever.py`: semantic retrieval for Memento-Text and StageWeaver banks.
- `scripts/run_stageweaver_eval.py`: thin canonical evaluation wrapper.
- `server/search_tool.py`: active SearXNG search MCP tool.
- `server/ai_crawl.py`: active crawl/extract MCP tool.
- `server/code_agent.py`: active workspace/code MCP tool.
- `server/documents_tool.py`: active document MCP tool.
- `server/image_tool.py`: active image MCP tool.
- `server/math_tool.py`: active math MCP tool.
- `server/excel_tool.py`: dependency of `documents_tool.py`.
- `server/interpreters/*`: dependency of `code_agent.py`.
- `infra/searxng/*`: local SearXNG Docker Compose deployment.
- `data/deepresearcher_protocol/seen_train.jsonl`: fixed DeepResearcher training split.
- `data/deepresearcher_protocol/seen_dev.jsonl`: fixed seen-dev gate split.
- `data/deepresearcher_protocol/ood_test.jsonl`: fixed OOD test split.
- `data/deepresearcher_protocol/protocol_split_stats.json`: documents the fixed split counts.
- `docs/CURRENT_PROTOCOL.md`: current protocol documentation.
- `docs/ARCHIVED_RESULTS_SUMMARY.md`: concise historical-results note.
- `README.md`: short entrypoint pointing to current docs.

Future workflow stages use semantic names, not historical R-run IDs. R001-R034 are historical diagnostics only. No R040+ naming is used in the current protocol.

Current recommended workflow outputs:

- `result/stageweaver/current/tracebank_seen_train/`
- `result/stageweaver/current/tracebank_seen_dev/`
- `result/stageweaver/current/stage_bank/`
- `result/stageweaver/current/append_sft/`
- `result/stageweaver/current/eval_seen_dev/`
- `result/stageweaver/current/eval_ood_test/`

## Legacy files to archive

- `archive/legacy_clients/agent.py`: older simplified hierarchical client superseded by `agent_local_server.py`.
- `archive/legacy_clients/no_parametric_cbr.py`: old CBR/Memento baseline runner outside the canonical runner.
- `archive/legacy_clients/parametric_memory_cbr.py`: old parametric CBR runner outside the canonical runner.
- `archive/legacy_memory/parametric_memory.py`: old parametric memory retriever utility.
- `archive/legacy_memory/np_memory.py`: old non-parametric memory retriever utility.
- `archive/legacy_memory/train_memory_retriever.py`: legacy BC/ranking retriever training path.
- `archive/legacy_memory/deepresearcher_protocol.py`: split-generation utility; active runner now consumes fixed splits directly.
- `archive/legacy_memory/training_data.jsonl`: old retriever training pairs.
- `archive/legacy_memory/memento_text_seed_memory.jsonl`: retired seed memory artifact previously stored at `memory/memory.jsonl`; current Memento-Text should rebuild from current-protocol planner traces via the StageWeaver bank.
- `archive/legacy_servers/serp_search.py`: old SerpAPI tool superseded by `search_tool.py`.
- `archive/legacy_servers/jina_fetch_tool.py`: optional Jina fetch tool outside the current six-tool profile.
- `archive/legacy_servers/craw_page.py`: old crawl-only tool superseded by `ai_crawl.py`.
- `archive/legacy_servers/video_tool.py`: video tool no longer default-loaded or imported by active tools.

## Legacy files safe to delete

- `memory/dummy_memo.jsonl`: tiny obsolete memory fixture not used by the active runner.
- `data/deepresearcher_protocol/seen_train_shards/*`: old parallel shard artifacts; active splits are consumed directly.
- `scripts/launch_vllm_6gpu.sh`: removed multi-endpoint launcher.
- `scripts/launch_parallel_traces.sh`: removed parallel shard runner.
- `scripts/merge_parallel_outputs.py`: removed shard output merger.
- `scripts/shard_jsonl.py`: removed split sharding helper.
- `scripts/__pycache__/*`: stale bytecode generated from removed scripts; safe to remove when filesystem permissions allow.
- `RUNNING_LOCAL.md`: replaced by focused docs under `docs/`.

## Files requiring manual review

- `.env`: local endpoint and API-key configuration; keep local and review before sharing.
- `uv.lock`: dependency lock retained because this cleanup did not reinstall dependencies.
- `client/code_tool.log`: generated log artifact; safe to remove if not needed for local debugging.
- `archive/**`: retained only for historical reference and must not be imported by active code.
