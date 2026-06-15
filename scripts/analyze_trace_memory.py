#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compact(value: Any, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 12, 1)].rstrip() + " [truncated]"


def _section(text: str, label: str, limit: int = 320) -> str:
    marker = f"[{label}]"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    next_match = re.search(r"\n\[[A-Z_]+\]", text[start:])
    end = start + next_match.start() if next_match else len(text)
    return _compact(text[start:end], limit)


def _select_trace(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    for row in rows:
        parts = str(row.get("task_id", "")).split("-")
        if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) == index:
            return row
    if 0 <= index < len(rows):
        return rows[index]
    raise IndexError(f"--index {index} not found in trace jsonl with {len(rows)} rows")


def _parse_plan(text: Any) -> list[dict[str, Any]]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    plan = parsed.get("plan")
    return plan if isinstance(plan, list) else []


def _tool_action(call: dict[str, Any]) -> str:
    name = call.get("resolved_name") or call.get("requested_name") or call.get("tool_name") or "unknown"
    args = call.get("arguments", call.get("arguments_raw", {}))
    if not isinstance(args, str):
        try:
            args = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except TypeError:
            args = str(args)
    status = "ERROR" if call.get("error") else "OK"
    reused = " reused" if call.get("reused_result") else ""
    return f"{name}({ _compact(args, 180) }) -> {status}{reused}"


def _bank_summary(case: dict[str, Any] | None) -> list[str]:
    if not case:
        return ["    - bank case: MISSING"]
    metadata = dict(case.get("metadata") or {})
    trajectory = dict(metadata.get("executor_trajectory") or {})
    lines = [
        f"    - stage/role/reward: `{case.get('stage')}` / `{case.get('agent_role')}` / `{case.get('reward')}`",
        f"    - task/state: {_compact(metadata.get('raw_task_description') or case.get('current_state_text') or case.get('state_text'), 220)}",
        f"    - target/output: {_compact(case.get('target_text'), 220)}",
    ]
    if case.get("stage") == "EXEC_STEP" or case.get("agent_role") == "executor":
        tool_calls = trajectory.get("tool_calls") or []
        lines.append(
            f"    - trajectory: legacy={trajectory.get('legacy_format', metadata.get('legacy_format'))}, tool_calls={len(tool_calls)}"
        )
        memory_text = metadata.get("executor_memory_text", "")
        if memory_text:
            lines.append(f"    - executor memory: {_compact(memory_text, 300)}")
    return lines


def _memory_block_md(title: str, mem: dict[str, Any], bank_by_id: dict[str, dict[str, Any]]) -> list[str]:
    lines = [f"#### {title}"]
    retrieval_query = str(mem.get("retrieval_query") or "")
    source_text = str(mem.get("source_text") or "")
    current_state = _section(retrieval_query, "EXECUTOR_TASK") or _section(source_text, "CURRENT_STATE") or _compact(retrieval_query, 260)
    latest_observation = _section(retrieval_query, "LATEST_OBSERVATION", 260) or _section(source_text, "LATEST_OBSERVATION", 260)
    ids = [str(x) for x in (mem.get("retrieved_ids") or [])]
    scores = mem.get("retrieved_scores") or []
    lines.extend(
        [
            f"- role/stage/mode/step: `{mem.get('role')}` / `{mem.get('stage')}` / `{mem.get('memory_refresh_mode')}` / `{mem.get('step_id')}`",
            f"- prefix tokens: `{mem.get('prefix_tokens', '')}`",
            f"- current_state: {_compact(current_state, 360)}",
            f"- latest_observation: {_compact(latest_observation, 260) or '[NONE]'}",
            f"- retrieval_query_hash: `{mem.get('retrieval_query_hash', '')}`",
            f"- source_text_hash: `{mem.get('source_text_hash', '')}`",
            f"- memory changed: `{mem.get('whether_memory_changed_from_previous_step', '')}`",
            f"- retrieved ids/scores: `{len(ids)}`",
        ]
    )
    for rank, source_id in enumerate(ids[:5], start=1):
        score = ""
        if rank - 1 < len(scores):
            try:
                score = f", score={float(scores[rank - 1]):.4f}"
            except (TypeError, ValueError):
                score = f", score={scores[rank - 1]}"
        lines.append(f"  - {rank}. `{source_id}`{score}")
        lines.extend(_bank_summary(bank_by_id.get(source_id)))
    if len(ids) > 5:
        lines.append(f"  - ... {len(ids) - 5} more")
    return lines


def build_markdown(trace: dict[str, Any], bank_by_id: dict[str, dict[str, Any]], index: int) -> str:
    lines = [
        f"# Trace Memory Debug: index {index}",
        "",
        f"- question: {_compact(trace.get('question'), 500)}",
        f"- final answer: {_compact(trace.get('final_answer'), 500)}",
        f"- trace error: {_compact(trace.get('error'), 300) or '[NONE]'}",
        f"- task_id: `{trace.get('task_id', '')}`",
        "",
    ]
    for cycle_idx, cycle in enumerate(trace.get("cycles") or []):
        lines.extend([f"## Cycle {cycle_idx}", ""])
        planner_output = str(cycle.get("planner_output") or "")
        lines.append(f"- planner output: {_compact(planner_output, 500)}")
        plan = _parse_plan(planner_output)
        if plan:
            lines.append("- planner tasks:")
            for task in plan:
                lines.append(f"  - Task {task.get('id')}: {_compact(task.get('description'), 220)}")
        else:
            lines.append("- planner tasks: [none parsed]")
        planner_mem = cycle.get("planner_memory")
        if isinstance(planner_mem, dict) and planner_mem:
            lines.extend(["", *_memory_block_md("Planner Memory", planner_mem, bank_by_id)])

        for task_idx, task_trace in enumerate(cycle.get("tasks") or []):
            task = dict(task_trace.get("task") or {})
            lines.extend(
                [
                    "",
                    f"### Executor Task {task.get('id', task_idx + 1)}",
                    f"- description: {_compact(task.get('description'), 360)}",
                    f"- final output: {_compact(task_trace.get('result'), 500)}",
                ]
            )
            calls = [call for call in (task_trace.get("tool_calls") or []) if isinstance(call, dict)]
            lines.append(f"- tool calls: `{len(calls)}`")
            for call in calls:
                lines.append(f"  - {_tool_action(call)}")
                if call.get("result_preview"):
                    lines.append(f"    - observation: {_compact(call.get('result_preview'), 280)}")
                if call.get("error"):
                    lines.append(f"    - error: {_compact(call.get('error'), 280)}")

            exec_mem = task_trace.get("executor_memory")
            if isinstance(exec_mem, dict) and exec_mem:
                lines.extend(["", *_memory_block_md("Executor Memory", exec_mem, bank_by_id)])
            steps = [step for step in (task_trace.get("executor_memory_steps") or []) if isinstance(step, dict)]
            if steps:
                lines.append("")
                lines.append(f"#### Executor Memory Steps: {len(steps)}")
                for step_idx, step in enumerate(steps, start=1):
                    lines.extend(_memory_block_md(f"Step {step_idx}", step, bank_by_id))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize StageWeaver runner trace memory hashes and retrieved cases.")
    parser.add_argument("--trace_jsonl", required=True)
    parser.add_argument("--bank_jsonl", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    trace_rows = _load_jsonl(Path(args.trace_jsonl))
    bank_rows = _load_jsonl(Path(args.bank_jsonl))
    bank_by_id = {str(row.get("source_id", "")): row for row in bank_rows if row.get("source_id")}
    trace = _select_trace(trace_rows, args.index)
    markdown = build_markdown(trace, bank_by_id, args.index)
    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output_md": str(output), "trace_rows": len(trace_rows), "bank_rows": len(bank_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
