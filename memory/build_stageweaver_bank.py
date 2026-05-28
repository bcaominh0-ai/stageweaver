from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

try:
    from .stageweaver_schema import (
        EXEC_STEP,
        PLAN_INIT,
        PLAN_REVISE,
        StageTuple,
        build_executor_current_state,
        save_stage_tuples,
    )
    from .stageweaver_serializers import serialize_executor_trajectory_case
except Exception:  # pragma: no cover
    from stageweaver_schema import EXEC_STEP, PLAN_INIT, PLAN_REVISE, StageTuple, build_executor_current_state, save_stage_tuples
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
                available_tools=[],
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


def build_tuples_from_traces(
    trace_rows: list[dict],
    result_rows: list[dict] | None = None,
    *,
    split_name: str = "unassigned",
) -> list[StageTuple]:
    reward_by_question = _reward_lookup(result_rows or [])
    info_by_question = _result_info_lookup(result_rows or [])
    tuples_: list[StageTuple] = []
    for trace_idx, trace in enumerate(trace_rows):
        question = str(trace.get("question", "")).strip()
        if not question:
            continue
        task_id = str(trace.get("task_id", f"trace-{trace_idx:05d}"))
        info = info_by_question.get(question, {})
        raw_reward = reward_by_question.get(question)
        if raw_reward is None:
            reward = 0
            success_signal = "unknown"
        else:
            reward = int(raw_reward)
            success_signal = "success" if reward == 1 else "failure"
        dataset_name = str(info.get("data_source", "")).strip() or "trace-bank"
        assigned_split = str(info.get("protocol_split", "")).strip() or split_name
        connected_tools = [str(x) for x in trace.get("connected_tools", [])]
        cycles = trace.get("cycles", [])
        for cycle_idx, cycle in enumerate(cycles):
            planner_output = str(cycle.get("planner_output", "")).strip()
            stage = _planner_stage(cycle_idx)
            if planner_output and not planner_output.startswith("FINAL ANSWER:"):
                tuples_.append(
                    StageTuple(
                        stage=stage,
                        episode_id=task_id,
                        query_id=task_id,
                        cycle_id=cycle_idx,
                        task_id=f"planner-cycle-{cycle_idx}",
                        state_text=_planner_state(trace, cycle_idx),
                        target_text=_planner_target(planner_output),
                        reward=reward,
                        dataset=dataset_name,
                        split=assigned_split,
                        source_id=f"{task_id}-planner-{cycle_idx}",
                        question_text=question,
                        agent_role="planner",
                        current_state_text=_planner_state(trace, cycle_idx),
                        tool_memory_text="\n".join(_tool_summary_lines(trace, cycle_idx, upto_task_idx=0)),
                        subtask_memory_text="\n".join(_task_result_lines(trace, cycle_idx, upto_task_idx=0)),
                        available_tools=connected_tools,
                        retrieved_ids=[],
                        metadata={"source_type": "trace", "trace_index": trace_idx},
                    )
                )
            for task_pos, task_trace in enumerate(cycle.get("tasks", [])):
                task_meta = task_trace.get("task", {})
                target_text = _executor_target(task_trace)
                current_state_text = str(task_meta.get("description", "")).strip()
                if not target_text or not current_state_text:
                    continue
                trajectory = _executor_trajectory_metadata(task_trace, current_state_text, reward, task_id, success_signal)
                executor_state_text = _executor_state_from_trajectory(current_state_text, trajectory)
                executor_metadata = {
                    "source_type": "trace",
                    "trace_index": trace_idx,
                    "task_description": current_state_text,
                    "raw_task_description": current_state_text,
                    "planner_output": planner_output,
                    "task_label": f"Task {task_meta.get('id', task_pos + 1)}: {current_state_text}",
                    "executor_trajectory": trajectory,
                    "legacy_format": trajectory["legacy_format"],
                    "trajectory_warnings": trajectory["warnings"],
                }
                executor_metadata["executor_memory_text"] = serialize_executor_trajectory_case(
                    {
                        "stage": EXEC_STEP,
                        "agent_role": "executor",
                        "state_text": executor_state_text,
                        "current_state_text": executor_state_text,
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
                        state_text=executor_state_text,
                        target_text=target_text,
                        reward=reward,
                        dataset=dataset_name,
                        split=assigned_split,
                        source_id=f"{task_id}-executor-{cycle_idx}-{task_pos}",
                        question_text=question,
                        agent_role="executor",
                        current_state_text=executor_state_text,
                        tool_memory_text="\n".join(_tool_summary_lines(trace, cycle_idx, upto_task_idx=task_pos)),
                        subtask_memory_text="\n".join(_task_result_lines(trace, cycle_idx, upto_task_idx=task_pos)),
                        available_tools=connected_tools,
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
        train = _load_split_tuples(args.train_trace_jsonl, args.train_results_jsonl, "train")
        val = _load_split_tuples(args.val_trace_jsonl, args.val_results_jsonl, "val") if args.val_trace_jsonl else []
        test = _load_split_tuples(args.test_trace_jsonl, args.test_results_jsonl, "test") if args.test_trace_jsonl else []
        tuples_ = train + val + test
    elif args.trace_jsonl:
        trace_path = Path(args.trace_jsonl)
        traces = load_jsonl(trace_path)
        results_rows = load_jsonl(Path(args.results_jsonl)) if args.results_jsonl and Path(args.results_jsonl).exists() else []
        tuples_ = build_tuples_from_traces(traces, result_rows=results_rows)
        train, val, test = assign_splits(tuples_, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
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
