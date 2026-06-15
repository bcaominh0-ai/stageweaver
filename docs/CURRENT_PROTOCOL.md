# Current Protocol

## Dataset

The active dataset protocol is DeepResearcher with three fixed JSONL files:

- `data/deepresearcher_protocol/seen_train.jsonl`
- `data/deepresearcher_protocol/seen_dev.jsonl`
- `data/deepresearcher_protocol/ood_test.jsonl`

Use `seen_dev` as the gate before reporting `ood_test`. The active runner consumes these split files directly; it does not regenerate or reshuffle them.

## Active Comparison

Only two research modes are active:

- `memento_text`: retrieves planner-stage text cases from the current StageWeaver bank and conditions the planner.
- `stageweaver`: retrieves role-scoped StageWeaver memory (`success_case` and `insight`) and converts it into latent memory.

`none` is diagnostic only for trace-bank construction and requires `--diagnostic_trace_bank`.

Current Memento-Text bootstrap rules:

- Memory source: `result/stageweaver/current/stage_bank/stage_bank_train.jsonl`
- Required prerequisite: run `build_stageweaver_bank` first
- Only planner tuples are used for Memento-Text retrieval memory
- Executor tuples are excluded
- There is no fallback to legacy `memory.jsonl`
- There is no fallback to `memento_text_seed_memory.jsonl`

## Runner

All evaluation flows through:

- `client/stageweaver_runner.py`
- optional wrapper: `scripts/run_stageweaver_eval.py`

## Semantic Workflow

Future workflow stages use semantic names only:

1. `collect_seen_train_traces`
2. `collect_seen_dev_traces`
3. `build_stageweaver_bank`
4. `train_append_sft`
5. `eval_seen_dev_gate`
6. `eval_ood_test`

If Linux helper scripts are added later, use semantic wrapper names such as `collect_seen_train_traces.sh`, `collect_seen_dev_traces.sh`, `build_stageweaver_bank.sh`, `train_append_sft.sh`, `eval_seen_dev_gate.sh`, and `eval_ood_test.sh`.

Recommended current output directories:

- `result/stageweaver/current/tracebank_seen_train/`
- `result/stageweaver/current/tracebank_seen_dev/`
- `result/stageweaver/current/stage_bank/`
- `result/stageweaver/current/append_sft/`
- `result/stageweaver/current/eval_seen_dev/`
- `result/stageweaver/current/eval_ood_test/`

`memento_text` should read planner tuples from `result/stageweaver/current/stage_bank/stage_bank_train.jsonl`. If that file is missing, run `build_stageweaver_bank` first. Legacy `memory/memory.jsonl` is not part of the current protocol.

## Tool Profile

The active tool profile contains exactly:

- `server/code_agent.py`
- `server/documents_tool.py`
- `server/image_tool.py`
- `server/math_tool.py`
- `server/ai_crawl.py`
- `server/search_tool.py`

`server/video_tool.py` is archived and is not default-loaded.

## StageWeaver

The active StageWeaver method is LatentMem-style role-conditioned composition:

- Source bank: complete agent traces are retained for audit, reproduction, and future distillation; they are not used directly for online retrieval.
- Role memory bank: online retrieval uses StageTuple rows scoped by `agent_role` and `stage`, with `metadata.memory_type` set to `success_case` or `insight`.
- Success cases come from successful traces. Failure insights are distilled from failed traces when `build_stageweaver_bank` is run with `--distill_failure_insights`; the LLM must assign each insight to the original failure stage (`PLAN_INIT`, `PLAN_REVISE`, or `EXEC_STEP`) and role (`planner` or `executor`).
- Input: `role + stage + current_state + retrieved role memory`
- Current state for retrieval is short and stage-stable: planner uses the original question for both `PLAN_INIT` and `PLAN_REVISE`; executor uses the current task description for `EXEC_STEP`.
- Interface: latent block appended after planner/executor prompt embeddings
- Required `--stage_mode both`
- Required `--latent_interface append`
- Required `--agent_backend local`
- No separate planner/executor text memory prompt in the latent branch
- No tool/subtask text memory in the latent branch
- Textual progress flows through normal shared history

Not active: GAIA, static/generic prefixes, Text-SI/Text-SB, planner-only latent runs, prepend latent interfaces, and legacy prompt bridges.
