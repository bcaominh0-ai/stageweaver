from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

PLAN_INIT = "PLAN_INIT"
PLAN_REVISE = "PLAN_REVISE"
EXEC_STEP = "EXEC_STEP"
VALID_STAGES = {PLAN_INIT, PLAN_REVISE, EXEC_STEP}


@dataclass
class StageTuple:
    stage: str
    episode_id: str
    query_id: str
    cycle_id: int
    task_id: str
    state_text: str
    target_text: str
    reward: int
    dataset: str
    split: str
    source_id: str
    question_text: str = ""
    agent_role: str = ""
    current_state_text: str = ""
    tool_memory_text: str = ""
    subtask_memory_text: str = ""
    available_tools: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if self.stage not in VALID_STAGES:
            raise ValueError(f"Unsupported stage: {self.stage}")
        data = asdict(self)
        data["reward"] = int(self.reward)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageTuple":
        fallback_seed = f"{data.get('stage','')}-{data.get('state_text','')}-{data.get('target_text','')}"
        fallback_id = hashlib.sha1(fallback_seed.encode("utf-8")).hexdigest()[:12]
        stage = str(data.get("stage", PLAN_INIT))
        if stage not in VALID_STAGES:
            raise ValueError(f"Unsupported stage in stage tuple: {stage}")
        return cls(
            stage=stage,
            episode_id=str(data.get("episode_id", f"legacy-episode-{fallback_id}")),
            query_id=str(data.get("query_id", f"legacy-query-{fallback_id}")),
            cycle_id=int(data.get("cycle_id", 0)),
            task_id=str(data.get("task_id", "")),
            state_text=str(data.get("state_text", "")),
            target_text=str(data.get("target_text", "")),
            reward=int(data.get("reward", 0)),
            dataset=str(data.get("dataset", "")),
            split=str(data.get("split", "")),
            source_id=str(data.get("source_id", f"legacy-source-{fallback_id}")),
            question_text=str(data.get("question_text", "")),
            agent_role=str(data.get("agent_role", "")),
            current_state_text=str(data.get("current_state_text", "")),
            tool_memory_text=str(data.get("tool_memory_text", "")),
            subtask_memory_text=str(data.get("subtask_memory_text", "")),
            available_tools=[str(x) for x in data.get("available_tools", [])],
            retrieved_ids=[str(x) for x in data.get("retrieved_ids", [])],
            metadata=dict(data.get("metadata", {})),
        )


def role_for_stage(stage: str) -> str:
    if stage == EXEC_STEP:
        return "executor"
    return "planner"


def tuple_question_text(item: StageTuple) -> str:
    return str(item.question_text or item.state_text or item.current_state_text).strip()


def tuple_current_state_text(item: StageTuple) -> str:
    return str(item.current_state_text or item.state_text).strip()


def tuple_tool_memory_text(item: StageTuple) -> str:
    return str(item.tool_memory_text).strip()


def tuple_subtask_memory_text(item: StageTuple) -> str:
    return str(item.subtask_memory_text).strip()


def tuple_role(item: StageTuple) -> str:
    return str(item.agent_role or role_for_stage(item.stage)).strip()


def normalize_optional_text(text: Any) -> str:
    normalized = str(text or "").strip()
    if normalized.lower() in {"none", "null"}:
        return ""
    return normalized


def _compact_line(text: Any, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[: max(max_chars - 12, 1)].rstrip() + " [truncated]"


def _tool_call_label(call: dict[str, Any]) -> str:
    tool_name = str(call.get("resolved_name") or call.get("requested_name") or call.get("tool_name") or "unknown")
    arguments = call.get("arguments", call.get("arguments_raw", call.get("args", {})))
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except TypeError:
            arguments = str(arguments)
    return f"{tool_name}({arguments})"


def build_executor_current_state(
    task_description: str,
    tool_history: list[dict[str, Any]] | None = None,
    latest_observation: str | None = None,
    failed_calls: list[Any] | None = None,
    repeated_calls: list[Any] | None = None,
    partial_result: str | None = None,
    *,
    max_chars: int = 4000,
    obs_max_chars: int = 1200,
    tool_history_k: int = 4,
) -> str:
    """Build an observation-aware executor retrieval state.

    The result is intentionally plain text so both text-memory and latent-memory
    paths can use exactly the same information.
    """
    history = list(tool_history or [])[-max(tool_history_k, 0) :] if tool_history_k else []
    latest = normalize_optional_text(latest_observation)
    if not latest and history:
        last_call = history[-1]
        latest = normalize_optional_text(last_call.get("result_preview") or last_call.get("observation") or last_call.get("error"))

    failure_lines: list[str] = []
    for item in failed_calls or []:
        if isinstance(item, dict):
            failure_lines.append(_compact_line(item.get("error") or item.get("message") or _tool_call_label(item), 240))
        else:
            failure_lines.append(_compact_line(item, 240))
    for item in repeated_calls or []:
        if isinstance(item, dict):
            failure_lines.append(f"repeated call detected: {_compact_line(_tool_call_label(item), 240)}")
        else:
            failure_lines.append(f"repeated call detected: {_compact_line(item, 240)}")

    parts: list[str] = [
        "[EXECUTOR_QUERY]",
        "[EXECUTOR_TASK]",
        _compact_line(task_description, 800) or "[NONE]",
        "",
        "[CALLED_TOOLS]",
    ]
    if history:
        for idx, call in enumerate(history, start=1):
            parts.append(f"{idx}. {_compact_line(_tool_call_label(call), 360)}")
    else:
        parts.append("[NONE]")

    parts.extend(
        [
            "",
            "[LATEST_OBSERVATION]",
            _compact_line(latest, obs_max_chars) or "[NONE]",
            "",
            "[FAILURE_OR_AMBIGUITY]",
        ]
    )
    if failure_lines:
        parts.extend(f"- {line}" for line in failure_lines if line)
    else:
        parts.append("[NONE]")

    parts.extend(
        [
            "",
            "[PARTIAL_RESULT]",
            _compact_line(partial_result, 800) or "[NONE]",
        ]
    )
    rendered = "\n".join(parts).strip()
    if max_chars > 0 and len(rendered) > max_chars:
        return rendered[: max(max_chars - 12, 1)].rstrip() + " [truncated]"
    return rendered


def looks_like_json_payload(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def retrieval_query_text(*, role: str, question: Any, current_state: Any) -> str:
    normalized_question = normalize_optional_text(question)
    normalized_state = normalize_optional_text(current_state)
    if role == "executor":
        return normalized_state or normalized_question
    return normalized_question


def serialize_role_conditioned_context(item: StageTuple, retrieved_cases_text: str = "") -> str:
    role = tuple_role(item)
    if role == "executor":
        current_state = retrieval_text(item)
    else:
        current_state = normalize_optional_text(tuple_current_state_text(item)) or "[NONE]"
    return serialize_role_conditioned_source(
        role=role,
        current_state=current_state,
        retrieved_cases_text=retrieved_cases_text,
    )


def serialize_role_conditioned_source(*, role: str, current_state: str, retrieved_cases_text: str = "") -> str:
    return "\n".join(
        [
            f"[ROLE] {role}",
            f"[CURRENT_STATE] {current_state}",
            "[RETRIEVED_POSITIVE_CASES]",
            retrieved_cases_text or "[NONE]",
        ]
    )


def stage_memory_retrieval_key(item: dict[str, Any]) -> str:
    role = str(item.get("agent_role", "")).strip()
    if role == "executor":
        state_text = str(item.get("state_text", "")).strip()
        if state_text:
            return state_text
    return retrieval_query_text(
        role=role,
        question=item.get("question", ""),
        current_state=item.get("current_state_text", item.get("state_text", "")),
    )


def require_positive_executor_cases(cases: list[dict[str, Any]], source: str | Path) -> None:
    if not cases:
        raise RuntimeError(
            "executor_memory_refresh=per_step requires positive executor trajectory cases. "
            f"No executor cases found in {source}."
        )


def require_explicit_executor_memory_jsonl(memory_mode: str, executor_memory_refresh: str, executor_memory_jsonl: str) -> None:
    if memory_mode == "memento_text" and executor_memory_refresh == "per_step" and not str(executor_memory_jsonl).strip():
        raise RuntimeError(
            "memento_text with executor_memory_refresh=per_step requires explicit "
            "--executor_memory_jsonl pointing to a StageTuple stage bank with executor trajectories."
        )


def retrieval_text(item: StageTuple) -> str:
    role = tuple_role(item)
    if role == "planner" and not normalize_optional_text(item.question_text):
        raise ValueError(f"planner retrieval requires question_text for source_id={item.source_id}")
    if role == "executor":
        materialized_state = normalize_optional_text(item.state_text or item.current_state_text)
        if materialized_state.startswith("[EXECUTOR_QUERY]"):
            return materialized_state
        trajectory = dict(item.metadata.get("executor_trajectory") or {})
        raw_task_description = str(
            item.metadata.get("raw_task_description")
            or item.metadata.get("task_description")
            or tuple_current_state_text(item)
        )
        tool_history = list(trajectory.get("tool_calls") or [])
        latest_observation = ""
        if tool_history:
            last_step = tool_history[-1]
            if isinstance(last_step, dict):
                latest_observation = str(last_step.get("observation_summary") or last_step.get("observation") or "")
        return build_executor_current_state(
            task_description=raw_task_description,
            tool_history=[dict(step) for step in tool_history if isinstance(step, dict)],
            latest_observation=latest_observation,
            failed_calls=[dict(step) for step in tool_history if isinstance(step, dict) and step.get("error")],
            repeated_calls=[dict(step) for step in tool_history if isinstance(step, dict) and step.get("reused_result")],
            partial_result=str(trajectory.get("final_output", "")),
            max_chars=2400,
            obs_max_chars=600,
            tool_history_k=4,
        )
    return retrieval_query_text(
        role=role,
        question=item.question_text if role == "planner" else tuple_question_text(item),
        current_state=tuple_current_state_text(item),
    )


def load_stage_tuples(path: str | Path) -> list[StageTuple]:
    items: list[StageTuple] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(StageTuple.from_dict(json.loads(line)))
    return items


def save_stage_tuples(path: str | Path, tuples_: Iterable[StageTuple]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in tuples_:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
