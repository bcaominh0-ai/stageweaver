from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

try:
    from .stageweaver_schema import (
        EXEC_STEP,
        PLAN_INIT,
        PLAN_REVISE,
        StageTuple,
        build_executor_current_state,
        is_role_memory_item,
        save_stage_tuples,
    )
    from .stageweaver_serializers import serialize_executor_trajectory_case
except Exception:  # pragma: no cover
    from stageweaver_schema import EXEC_STEP, PLAN_INIT, PLAN_REVISE, StageTuple, build_executor_current_state, is_role_memory_item, save_stage_tuples
    from stageweaver_serializers import serialize_executor_trajectory_case


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_memory_entries(path: Path) -> list[dict]:
    return load_jsonl(path)


def reward_from_entry(entry: dict) -> int:
    if "reward" in entry:
        return int(entry["reward"])
    label = str(entry.get("case_label", "")).lower().strip()
    return 1 if label == "positive" else 0


def build_legacy_memory_tuples(rows: list[dict]) -> list[StageTuple]:
    tuples_: list[StageTuple] = []
    for idx, row in enumerate(rows):
        question = str(row.get("question", row.get("case", ""))).strip()
        plan = str(row.get("plan", "")).strip()
        if not question or not plan:
            continue
        tuples_.append(
            StageTuple(
                stage=PLAN_INIT,
                episode_id=f"memory-episode-{idx:05d}",
                query_id=f"memory-query-{idx:05d}",
                cycle_id=0,
                task_id="planner",
                state_text=question,
                target_text=plan,
                reward=reward_from_entry(row),
                dataset="memory-bank",
                split="unassigned",
                source_id=f"memory-{idx:05d}",
                question_text=question,
                agent_role="planner",
                current_state_text=question,
                tool_memory_text="",
                subtask_memory_text="",
                retrieved_ids=[],
                metadata={"case_label": row.get("case_label", ""), "raw_index": idx, "source_type": "legacy-memory"},
            )
        )
    return tuples_


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _task_result_lines(trace: dict, cycle_idx: int, upto_task_idx: int | None = None, limit: int = 6) -> list[str]:
    rows: list[str] = []
    for c_idx, cycle in enumerate(trace.get("cycles", [])):
        if c_idx > cycle_idx:
            break
        tasks = cycle.get("tasks", [])
        for t_idx, task in enumerate(tasks):
            if c_idx == cycle_idx and upto_task_idx is not None and t_idx >= upto_task_idx:
                break
            result = _normalize_text(str(task.get("result", "")))
            task_meta = task.get("task", {})
            task_id = task_meta.get("id", t_idx + 1)
            if result:
                rows.append(f"Task {task_id} result: {result}")
    return rows[-limit:]


def _tool_summary_lines(trace: dict, cycle_idx: int, upto_task_idx: int | None = None, limit: int = 6) -> list[str]:
    rows: list[str] = []
    for c_idx, cycle in enumerate(trace.get("cycles", [])):
        if c_idx > cycle_idx:
            break
        tasks = cycle.get("tasks", [])
        for t_idx, task in enumerate(tasks):
            if c_idx == cycle_idx and upto_task_idx is not None and t_idx >= upto_task_idx:
                break
            for tool_call in task.get("tool_calls", []):
                tool_name = str(tool_call.get("resolved_name") or tool_call.get("requested_name") or "unknown")
                arguments = tool_call.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                result_preview = _normalize_text(str(tool_call.get("result_preview", "")))
                rows.append(f"{tool_name}({arguments}) -> {result_preview or '[no preview]'}")
    return rows[-limit:]


def _planner_stage(cycle_idx: int) -> str:
    return PLAN_INIT if cycle_idx == 0 else PLAN_REVISE


def _planner_state(trace: dict, cycle_idx: int) -> str:
    if cycle_idx == 0:
        return str(trace.get("question", "")).strip()
    prev_cycles = trace.get("cycles", [])
    if cycle_idx - 1 < len(prev_cycles):
        prev_output = str(prev_cycles[cycle_idx - 1].get("planner_output", "")).strip()
        if prev_output:
            return prev_output
    return str(trace.get("question", "")).strip()


def _planner_target(planner_output: str) -> str:
    return planner_output.strip()


def _is_planner_final_output(planner_output: str) -> bool:
    text = planner_output.strip()
    if text.startswith("FINAL ANSWER:"):
        return True
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text.startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and set(payload.keys()) == {"final"}


def _executor_target(task_trace: dict) -> str:
    tool_calls = task_trace.get("tool_calls") or []
    if tool_calls:
        first_call = tool_calls[0]
        tool_name = str(first_call.get("resolved_name") or first_call.get("requested_name") or "unknown")
        arguments = first_call.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        return f"[TOOL_CALL] {tool_name}({arguments})"
    result = _normalize_text(str(task_trace.get("result", "")))
    return f"[RETURN] {result}" if result else ""


def _truncate_text(text: str, limit: int = 700) -> str:
    clean = _normalize_text(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(limit - 12, 1)].rstrip() + " [truncated]"


def _parse_json_object(text: str) -> dict[str, Any]:
    clean = str(text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[^\n]*\n", "", clean)
        clean = re.sub(r"\n?```$", "", clean).strip()
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM insight response must be a JSON object")
    return payload


def _tool_call_string(tool_call: dict) -> str:
    tool_name = str(tool_call.get("resolved_name") or tool_call.get("requested_name") or "unknown")
    arguments = tool_call.get("arguments", tool_call.get("arguments_raw", {}))
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except TypeError:
            arguments = str(arguments)
    return f"{tool_name}({arguments})"


def _decision_rationale(tool_call: dict, *, is_last: bool) -> str:
    if tool_call.get("error"):
        return "The tool call failed or returned unusable output, so the next executor action should repair the call or change strategy."
    if tool_call.get("reused_result"):
        return "The same tool call had already been executed; use the cached observation instead of repeating it."
    if is_last:
        return "The observation was used to produce the task result or decide that enough evidence had been collected."
    return "The observation informs whether to refine the query, call another tool, crawl a result, or finish."


def _executor_trajectory_metadata(
    task_trace: dict,
    task_description: str,
    reward: int,
    task_id: str,
    success_signal: str,
) -> dict:
    warnings: list[str] = []
    tool_steps: list[dict] = []
    tool_calls = task_trace.get("tool_calls") or []
    if tool_calls and not isinstance(tool_calls, list):
        warnings.append("tool_calls was not a list")
        tool_calls = []
    for idx, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            warnings.append(f"tool_call_{idx} was not an object")
            continue
        observation = str(call.get("result_preview") or call.get("error") or "")
        tool_steps.append(
            {
                "step_id": idx + 1,
                "tool_name": str(call.get("resolved_name") or call.get("requested_name") or "unknown"),
                "arguments": call.get("arguments", call.get("arguments_raw", {})),
                "tool_call": _tool_call_string(call),
                "observation": _truncate_text(observation, 1000),
                "observation_summary": _truncate_text(observation, 500),
                "decision_rationale": _decision_rationale(call, is_last=idx == len(tool_calls) - 1),
                "error": str(call.get("error", "")),
                "reused_result": bool(call.get("reused_result", False)),
            }
        )
    final_output = _normalize_text(str(task_trace.get("result", "")))
    legacy_format = not bool(tool_steps)
    if legacy_format:
        warnings.append("executor task has no parsed tool trajectory; falling back to output-only memory")
    return {
        "task_description": task_description,
        "executor_role": "executor",
        "tool_calls": tool_steps,
        "observations": [step["observation"] for step in tool_steps if step.get("observation")],
        "observation_summaries": [step["observation_summary"] for step in tool_steps if step.get("observation_summary")],
        "decision_rationales": [step["decision_rationale"] for step in tool_steps if step.get("decision_rationale")],
        "final_output": final_output,
        "success_signal": success_signal,
        "is_success": bool(int(reward) == 1),
        "source_trace_id": task_id,
        "legacy_format": legacy_format,
        "warnings": warnings,
    }


def _executor_state_from_trajectory(task_description: str, trajectory: dict) -> str:
    tool_history = [dict(step) for step in trajectory.get("tool_calls", []) if isinstance(step, dict)]
    latest_observation = ""
    if tool_history:
        latest = tool_history[-1]
        latest_observation = str(latest.get("observation_summary") or latest.get("observation") or latest.get("error") or "")
    return build_executor_current_state(
        task_description=task_description,
        tool_history=tool_history,
        latest_observation=latest_observation,
        failed_calls=[step for step in tool_history if step.get("error")],
        repeated_calls=[step for step in tool_history if step.get("reused_result")],
        partial_result=str(trajectory.get("final_output", "")),
    )


def _failure_trace_prompt(trace: dict, trace_id: str, max_chars: int = 9000) -> str:
    question = str(trace.get("question", "")).strip()
    lines: list[str] = [
        "You are an advanced reasoning agent that derives reusable StageWeaver insights from failed multi-agent traces.",
        "Return only JSON with this shape:",
        '{"insights":[{"agent_role":"planner|executor","stage":"PLAN_INIT|PLAN_REVISE|EXEC_STEP","current_state":"...","insight":"..."}]}',
        "",
        "You are given one failed trace. Identify where the decisive failure was introduced, then extract concise, generally applicable insights that help avoid similar failures in future tasks.",
        "",
        "Attribution:",
        "- Use PLAN_INIT if the failure comes from the initial decomposition, missing constraints, or wrong search direction.",
        "- Use PLAN_REVISE if the failure comes from poor adaptation after partial evidence, premature finalization, or failure to repair the plan.",
        "- Use EXEC_STEP if the failure comes from tool use, search/crawl strategy, evidence extraction, observation interpretation, or verification.",
        "- Attribute each insight to the earliest stage where changing behavior would likely have prevented the failure.",
        "- Do not invent a new stage; planner insights use agent_role=planner and executor insights use agent_role=executor.",
        "- For planner current_state, use the original question; for executor current_state, use the failed executor task description.",
        "",
        "Insight quality:",
        "- Infer what a successful trajectory would have done differently.",
        "- Focus on the causal difference between effective and ineffective behavior, not on surface details of this task.",
        "- Make each insight concise, high-level, and reusable.",
        "- Do not restate task-specific details.",
        "- Say what to avoid and what to do instead.",
        "- Prefer 1-3 insights.",
        "",
        f"[TRACE_ID] {trace_id}",
        f"[QUESTION] {question}",
        "[CYCLES]",
    ]
    for cycle_idx, cycle in enumerate(trace.get("cycles", [])):
        planner_output = _truncate_text(str(cycle.get("planner_output", "")), 1600)
        lines.extend(
            [
                f"Cycle {cycle_idx} stage={_planner_stage(cycle_idx)}",
                f"Planner output: {planner_output or '[NONE]'}",
            ]
        )
        for task_pos, task_trace in enumerate(cycle.get("tasks", [])):
            task_meta = task_trace.get("task", {})
            task_desc = _truncate_text(str(task_meta.get("description", "")), 900)
            result = _truncate_text(str(task_trace.get("result", "")), 900)
            lines.append(f"Task {task_pos + 1}: {task_desc or '[NONE]'}")
            for call_idx, call in enumerate(task_trace.get("tool_calls") or [], start=1):
                if not isinstance(call, dict):
                    continue
                call_text = _truncate_text(_tool_call_string(call), 700)
                obs = _truncate_text(str(call.get("result_preview") or call.get("error") or ""), 700)
                lines.append(f"  Tool {call_idx}: {call_text} -> {obs or '[NONE]'}")
            lines.append(f"  Result: {result or '[NONE]'}")
    rendered = "\n".join(lines).strip()
    if len(rendered) > max_chars:
        return rendered[: max(max_chars - 12, 1)].rstrip() + " [truncated]"
    return rendered


def _call_openai_insight_llm(prompt: str, *, model: str, api_key: str, base_url: str = "") -> str:
    if not api_key:
        raise RuntimeError("Failure insight distillation requires --insight_api_key or INSIGHT_API_KEY.")
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Failure insight distillation requires the openai package.") from exc
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You distill failed agent traces into compact, stage-attributed StageWeaver insights. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return str(response.choices[0].message.content or "")


def _valid_insight_stage(agent_role: str, stage: str) -> bool:
    if agent_role == "planner":
        return stage in {PLAN_INIT, PLAN_REVISE}
    if agent_role == "executor":
        return stage == EXEC_STEP
    return False


def _insight_current_state(raw: dict[str, Any], *, role: str, question: str, trace: dict) -> str:
    explicit = str(raw.get("current_state") or raw.get("task_description") or "").strip()
    if role == "planner":
        return question
    if explicit:
        return explicit
    for cycle in trace.get("cycles", []):
        for task_trace in cycle.get("tasks", []):
            task_meta = task_trace.get("task", {})
            task_desc = str(task_meta.get("description", "")).strip()
            if task_desc:
                return task_desc
    return question


def distill_failure_insight_tuples(
    trace_rows: list[dict],
    result_rows: list[dict] | None = None,
    *,
    split_name: str = "unassigned",
    model: str = "gpt-4.1-mini",
    api_key: str = "",
    base_url: str = "",
    max_traces: int = 0,
    insights_per_trace: int = 3,
    llm_fn: Callable[[str], str] | None = None,
) -> list[StageTuple]:
    reward_by_question = _reward_lookup(result_rows or [])
    info_by_question = _result_info_lookup(result_rows or [])
    tuples_: list[StageTuple] = []
    processed = 0
    for trace_idx, trace in enumerate(trace_rows):
        question = str(trace.get("question", "")).strip()
        if not question or reward_by_question.get(question) != 0:
            continue
        if max_traces > 0 and processed >= max_traces:
            break
        processed += 1
        trace_id = str(trace.get("task_id", f"trace-{trace_idx:05d}"))
        prompt = _failure_trace_prompt(trace, trace_id)
        raw_response = llm_fn(prompt) if llm_fn is not None else _call_openai_insight_llm(prompt, model=model, api_key=api_key, base_url=base_url)
        payload = _parse_json_object(raw_response)
        raw_insights = payload.get("insights", [])
        if not isinstance(raw_insights, list):
            continue
        info = info_by_question.get(question, {})
        dataset_name = str(info.get("data_source", "")).strip() or "trace-bank"
        assigned_split = str(info.get("protocol_split", "")).strip() or split_name
        kept_for_trace = 0
        for raw_idx, raw in enumerate(raw_insights):
            if kept_for_trace >= insights_per_trace:
                break
            if not isinstance(raw, dict):
                continue
            role = str(raw.get("agent_role", "")).strip()
            stage = str(raw.get("stage", "")).strip()
            insight = _normalize_text(str(raw.get("insight", "")))
            if not insight or not _valid_insight_stage(role, stage):
                continue
            state_text = _insight_current_state(raw, role=role, question=question, trace=trace)
            source_id = f"{trace_id}-insight-{kept_for_trace}"
            tuples_.append(
                StageTuple(
                    stage=stage,
                    episode_id=trace_id,
                    query_id=trace_id,
                    cycle_id=0,
                    task_id=f"{role}-insight-{raw_idx}",
                    state_text=state_text,
                    target_text=insight,
                    reward=0,
                    dataset=dataset_name,
                    split=assigned_split,
                    source_id=source_id,
                    question_text=question,
                    agent_role=role,
                    current_state_text=state_text,
                    metadata={
                        "memory_type": "insight",
                        "origin": "failure",
                        "source_ids": [trace_id],
                        "source_type": "llm_failure_distillation",
                        "trace_index": trace_idx,
                    },
                )
            )
            kept_for_trace += 1
    return tuples_


def _reward_lookup(results_rows: list[dict]) -> dict[str, int | None]:
    reward_by_question: dict[str, int | None] = {}
    for row in results_rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        reward_by_question[question] = int(bool(row["correct"])) if "correct" in row else None
    return reward_by_question


def _result_info_lookup(results_rows: list[dict]) -> dict[str, dict[str, str | int]]:
    info_by_question: dict[str, dict[str, str | int]] = {}
    for row in results_rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        info_by_question[question] = {
            "reward": int(bool(row["correct"])) if "correct" in row else "unknown",
            "data_source": str(row.get("data_source", "")).strip(),
            "protocol_split": str(row.get("protocol_split", "")).strip(),
        }
    return info_by_question


def _trace_sample_index(trace: dict) -> int | None:
    match = re.match(r"^[^-]+-(\d+)-", str(trace.get("task_id", "")))
    return int(match.group(1)) if match else None


def _matched_result_info(trace_rows: list[dict], result_rows: list[dict]) -> list[dict[str, Any]]:
    """Match retries without collapsing duplicate questions or reordered rows."""
    buckets: dict[tuple[int, str], deque[dict]] = defaultdict(deque)
    by_question: dict[str, list[dict]] = defaultdict(list)
    for row in result_rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        sample_index = row.get("source_sample_index")
        if sample_index is None:
            sample_index = row.get("index")
        if sample_index is not None:
            buckets[(int(sample_index), question)].append(row)
        by_question[question].append(row)

    matched: list[dict[str, Any]] = []
    for trace in trace_rows:
        question = str(trace.get("question", "")).strip()
        sample_index = _trace_sample_index(trace)
        row = None
        if sample_index is not None:
            queue = buckets.get((sample_index, question))
            if queue:
                row = queue.popleft()
        if row is None and len(by_question.get(question, [])) == 1:
            row = by_question[question][0]
        matched.append(
            {
                "reward": int(bool(row["correct"])) if row is not None and "correct" in row else None,
                "data_source": str((row or {}).get("data_source", "")).strip(),
                "protocol_split": str((row or {}).get("protocol_split", "")).strip(),
            }
        )
    return matched


def build_tuples_from_traces(
    trace_rows: list[dict],
    result_rows: list[dict] | None = None,
    *,
    split_name: str = "unassigned",
) -> list[StageTuple]:
    matched_info = _matched_result_info(trace_rows, result_rows or [])
    tuples_: list[StageTuple] = []
    for trace_idx, trace in enumerate(trace_rows):
        question = str(trace.get("question", "")).strip()
        if not question:
            continue
        task_id = str(trace.get("task_id", f"trace-{trace_idx:05d}"))
        info = matched_info[trace_idx]
        raw_reward = info.get("reward")
        if raw_reward is None:
            reward = 0
            success_signal = "unknown"
        else:
            reward = int(raw_reward)
            success_signal = "success" if reward == 1 else "failure"
        dataset_name = str(info.get("data_source", "")).strip() or "trace-bank"
        assigned_split = str(info.get("protocol_split", "")).strip() or split_name
        cycles = trace.get("cycles", [])
        for cycle_idx, cycle in enumerate(cycles):
            planner_output = str(cycle.get("planner_output", "")).strip()
            stage = _planner_stage(cycle_idx)
            if planner_output and not _is_planner_final_output(planner_output):
                planner_source_id = f"{task_id}-planner-{cycle_idx}"
                planner_metadata = {"source_type": "trace", "trace_index": trace_idx, "trace_id": task_id}
                if reward == 1:
                    planner_metadata.update(
                        {
                            "memory_type": "success_case",
                            "origin": "success",
                            "source_ids": [planner_source_id],
                        }
                    )
                tuples_.append(
                    StageTuple(
                        stage=stage,
                        episode_id=task_id,
                        query_id=task_id,
                        cycle_id=cycle_idx,
                        task_id=f"planner-cycle-{cycle_idx}",
                        state_text=question,
                        target_text=_planner_target(planner_output),
                        reward=reward,
                        dataset=dataset_name,
                        split=assigned_split,
                        source_id=planner_source_id,
                        question_text=question,
                        agent_role="planner",
                        current_state_text=question,
                        tool_memory_text="\n".join(_tool_summary_lines(trace, cycle_idx, upto_task_idx=0)),
                        subtask_memory_text="\n".join(_task_result_lines(trace, cycle_idx, upto_task_idx=0)),
                        retrieved_ids=[],
                        metadata=planner_metadata,
                    )
                )
            for task_pos, task_trace in enumerate(cycle.get("tasks", [])):
                task_meta = task_trace.get("task", {})
                target_text = _executor_target(task_trace)
                current_state_text = str(task_meta.get("description", "")).strip()
                if not target_text or not current_state_text:
                    continue
                trajectory = _executor_trajectory_metadata(task_trace, current_state_text, reward, task_id, success_signal)
                executor_metadata = {
                    "source_type": "trace",
                    "trace_index": trace_idx,
                    "trace_id": task_id,
                    "task_description": current_state_text,
                    "raw_task_description": current_state_text,
                    "planner_output": planner_output,
                    "task_label": f"Task {task_meta.get('id', task_pos + 1)}: {current_state_text}",
                    "executor_trajectory": trajectory,
                    "legacy_format": trajectory["legacy_format"],
                    "trajectory_warnings": trajectory["warnings"],
                }
                executor_source_id = f"{task_id}-executor-{cycle_idx}-{task_pos}"
                if reward == 1:
                    executor_metadata.update(
                        {
                            "memory_type": "success_case",
                            "origin": "success",
                            "source_ids": [executor_source_id],
                        }
                    )
                executor_metadata["executor_memory_text"] = serialize_executor_trajectory_case(
                    {
                        "stage": EXEC_STEP,
                        "agent_role": "executor",
                        "state_text": current_state_text,
                        "current_state_text": current_state_text,
                        "target_text": target_text,
                        "reward": reward,
                        "metadata": executor_metadata,
                    }
                )
                tuples_.append(
                    StageTuple(
                        stage=EXEC_STEP,
                        episode_id=task_id,
                        query_id=task_id,
                        cycle_id=cycle_idx,
                        task_id=f"executor-task-{task_meta.get('id', task_pos + 1)}",
                        state_text=current_state_text,
                        target_text=target_text,
                        reward=reward,
                        dataset=dataset_name,
                        split=assigned_split,
                        source_id=executor_source_id,
                        question_text=question,
                        agent_role="executor",
                        current_state_text=current_state_text,
                        tool_memory_text="\n".join(_tool_summary_lines(trace, cycle_idx, upto_task_idx=task_pos)),
                        subtask_memory_text="\n".join(_task_result_lines(trace, cycle_idx, upto_task_idx=task_pos)),
                        retrieved_ids=[],
                        metadata=executor_metadata,
                    )
                )
    return tuples_


def assign_splits(items: list[StageTuple], seed: int, train_frac: float, val_frac: float) -> tuple[list[StageTuple], list[StageTuple], list[StageTuple]]:
    rng = random.Random(seed)
    grouped: dict[str, list[StageTuple]] = {}
    for item in items:
        grouped.setdefault(item.episode_id, []).append(item)
    episode_ids = list(grouped.keys())
    rng.shuffle(episode_ids)
    n = len(episode_ids)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train_ids = set(episode_ids[:train_end])
    val_ids = set(episode_ids[train_end:val_end])
    test_ids = set(episode_ids[val_end:])
    train = [item for item in items if item.episode_id in train_ids]
    val = [item for item in items if item.episode_id in val_ids]
    test = [item for item in items if item.episode_id in test_ids]
    for split_name, split_items in (("train", train), ("val", val), ("test", test)):
        for item in split_items:
            item.split = split_name
    return train, val, test


def dump_stats(path: Path, train: list[StageTuple], val: list[StageTuple], test: list[StageTuple]) -> None:
    def _count(items: list[StageTuple], stage: str) -> int:
        return sum(1 for item in items if item.stage == stage)

    stats = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "train_positive": sum(item.reward for item in train),
        "val_positive": sum(item.reward for item in val),
        "test_positive": sum(item.reward for item in test),
        "train_plan_init": _count(train, PLAN_INIT),
        "train_plan_revise": _count(train, PLAN_REVISE),
        "train_exec_step": _count(train, EXEC_STEP),
        "val_plan_init": _count(val, PLAN_INIT),
        "val_plan_revise": _count(val, PLAN_REVISE),
        "val_exec_step": _count(val, EXEC_STEP),
        "test_plan_init": _count(test, PLAN_INIT),
        "test_plan_revise": _count(test, PLAN_REVISE),
        "test_exec_step": _count(test, EXEC_STEP),
    }
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_split_tuples(trace_jsonl: str, results_jsonl: str, split_name: str) -> list[StageTuple]:
    trace_path = Path(trace_jsonl)
    if not trace_path.exists():
        raise FileNotFoundError(f"{split_name} trace source missing: {trace_path}")
    trace_rows = load_jsonl(trace_path)
    results_rows = load_jsonl(Path(results_jsonl)) if results_jsonl and Path(results_jsonl).exists() else []
    tuples_ = build_tuples_from_traces(trace_rows, result_rows=results_rows, split_name=split_name)
    for item in tuples_:
        item.split = split_name
    return tuples_


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory_jsonl",
        type=Path,
        default=Path(""),
        help="Retired legacy input. Current protocol builds the stage bank from collected traces, not from legacy memory JSONL files.",
    )
    parser.add_argument("--trace_jsonl", type=str, default="")
    parser.add_argument("--results_jsonl", type=str, default="")
    parser.add_argument("--train_trace_jsonl", type=str, default="")
    parser.add_argument("--train_results_jsonl", type=str, default="")
    parser.add_argument("--val_trace_jsonl", type=str, default="")
    parser.add_argument("--val_results_jsonl", type=str, default="")
    parser.add_argument("--test_trace_jsonl", type=str, default="")
    parser.add_argument("--test_results_jsonl", type=str, default="")
    parser.add_argument("--output_dir", type=Path, default=Path("result/stageweaver/current/stage_bank"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--small_train_size", type=int, default=200)
    parser.add_argument("--small_val_size", type=int, default=32)
    parser.add_argument(
        "--val_frac_from_train",
        type=float,
        default=0.0,
        help="Move this fraction of train episodes into validation after tuple construction.",
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="Keep only StageTuples derived from successful traces (reward == 1).",
    )
    parser.add_argument(
        "--distill_failure_insights",
        action="store_true",
        help="Use an LLM to distill failed traces into stage-attributed insight role-memory rows.",
    )
    parser.add_argument("--insight_model", type=str, default=os.environ.get("INSIGHT_MODEL", ""))
    parser.add_argument("--insight_api_key", type=str, default=os.environ.get("INSIGHT_API_KEY", ""))
    parser.add_argument("--insight_base_url", type=str, default=os.environ.get("INSIGHT_BASE_URL", ""))
    parser.add_argument("--insight_max_traces", type=int, default=0, help="Maximum failed traces to distill; 0 means all.")
    parser.add_argument("--insights_per_trace", type=int, default=3)
    args = parser.parse_args()

    explicit_split_mode = any(
        (
            args.train_trace_jsonl,
            args.val_trace_jsonl,
            args.test_trace_jsonl,
        )
    )
    if explicit_split_mode:
        if not args.train_trace_jsonl:
            raise SystemExit("explicit split mode requires --train_trace_jsonl")
        train_trace_rows = load_jsonl(Path(args.train_trace_jsonl))
        train_result_rows = load_jsonl(Path(args.train_results_jsonl)) if args.train_results_jsonl and Path(args.train_results_jsonl).exists() else []
        train = build_tuples_from_traces(train_trace_rows, result_rows=train_result_rows, split_name="train")
        val_trace_rows = load_jsonl(Path(args.val_trace_jsonl)) if args.val_trace_jsonl else []
        val_result_rows = load_jsonl(Path(args.val_results_jsonl)) if args.val_results_jsonl and Path(args.val_results_jsonl).exists() else []
        val = build_tuples_from_traces(val_trace_rows, result_rows=val_result_rows, split_name="val") if val_trace_rows else []
        test_trace_rows = load_jsonl(Path(args.test_trace_jsonl)) if args.test_trace_jsonl else []
        test_result_rows = load_jsonl(Path(args.test_results_jsonl)) if args.test_results_jsonl and Path(args.test_results_jsonl).exists() else []
        test = build_tuples_from_traces(test_trace_rows, result_rows=test_result_rows, split_name="test") if test_trace_rows else []
        if args.distill_failure_insights:
            train.extend(
                distill_failure_insight_tuples(
                    train_trace_rows,
                    result_rows=train_result_rows,
                    split_name="train",
                    model=args.insight_model,
                    api_key=args.insight_api_key,
                    base_url=args.insight_base_url,
                    max_traces=args.insight_max_traces,
                    insights_per_trace=args.insights_per_trace,
                )
            )
            if val_trace_rows:
                val.extend(
                    distill_failure_insight_tuples(
                        val_trace_rows,
                        result_rows=val_result_rows,
                        split_name="val",
                        model=args.insight_model,
                        api_key=args.insight_api_key,
                        base_url=args.insight_base_url,
                        max_traces=args.insight_max_traces,
                        insights_per_trace=args.insights_per_trace,
                    )
                )
            if test_trace_rows:
                test.extend(
                    distill_failure_insight_tuples(
                        test_trace_rows,
                        result_rows=test_result_rows,
                        split_name="test",
                        model=args.insight_model,
                        api_key=args.insight_api_key,
                        base_url=args.insight_base_url,
                        max_traces=args.insight_max_traces,
                        insights_per_trace=args.insights_per_trace,
                    )
                )
        for item in train:
            item.split = "train"
        for item in val:
            item.split = "val"
        for item in test:
            item.split = "test"
        train = [item for item in train if is_role_memory_item(item)]
        val = [item for item in val if is_role_memory_item(item)]
        test = [item for item in test if is_role_memory_item(item)]
        if args.success_only:
            train = [item for item in train if item.reward == 1]
            val = [item for item in val if item.reward == 1]
            test = [item for item in test if item.reward == 1]
        if args.val_frac_from_train:
            if not 0.0 < args.val_frac_from_train < 1.0:
                raise SystemExit("--val_frac_from_train must be between 0 and 1")
            if val:
                raise SystemExit("--val_frac_from_train requires an empty explicit validation split")
            episode_ids = sorted({item.episode_id for item in train})
            random.Random(args.seed).shuffle(episode_ids)
            val_count = max(1, int(len(episode_ids) * args.val_frac_from_train))
            val_ids = set(episode_ids[:val_count])
            val = [item for item in train if item.episode_id in val_ids]
            train = [item for item in train if item.episode_id not in val_ids]
            for item in train:
                item.split = "train"
            for item in val:
                item.split = "val"
        tuples_ = train + val + test
    elif args.trace_jsonl:
        trace_path = Path(args.trace_jsonl)
        traces = load_jsonl(trace_path)
        results_rows = load_jsonl(Path(args.results_jsonl)) if args.results_jsonl and Path(args.results_jsonl).exists() else []
        tuples_ = build_tuples_from_traces(traces, result_rows=results_rows)
        if args.distill_failure_insights:
            tuples_.extend(
                distill_failure_insight_tuples(
                    traces,
                    result_rows=results_rows,
                    model=args.insight_model,
                    api_key=args.insight_api_key,
                    base_url=args.insight_base_url,
                    max_traces=args.insight_max_traces,
                    insights_per_trace=args.insights_per_trace,
                )
            )
        if args.success_only:
            tuples_ = [item for item in tuples_ if item.reward == 1]
        train, val, test = assign_splits(tuples_, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
        train = [item for item in train if is_role_memory_item(item)]
        val = [item for item in val if is_role_memory_item(item)]
        test = [item for item in test if is_role_memory_item(item)]
        tuples_ = train + val + test
    else:
        raise SystemExit(
            "Current protocol requires --trace_jsonl or explicit --train_trace_jsonl/--val_trace_jsonl/--test_trace_jsonl inputs. "
            "Legacy memory_jsonl stage-bank construction has been retired."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_stage_tuples(args.output_dir / "stage_bank_train.jsonl", train)
    save_stage_tuples(args.output_dir / "stage_bank_val.jsonl", val)
    save_stage_tuples(args.output_dir / "stage_bank_test.jsonl", test)
    save_stage_tuples(args.output_dir / "stage_bank_train_small.jsonl", train[: args.small_train_size])
    save_stage_tuples(args.output_dir / "stage_bank_val_small.jsonl", val[: args.small_val_size])
    dump_stats(args.output_dir / "stage_bank_stats.json", train, val, test)
    print(json.dumps({"output_dir": str(args.output_dir), "tuples": len(tuples_)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
