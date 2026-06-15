from __future__ import annotations

import hashlib
import json
import re
from typing import Any

try:
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - smoke-test fallback
    tiktoken = None


POSITIVE_OUTPUT_FIELD_ORDER = [
    ("output", "[OUTPUT]"),
]
SUCCESS_CASE = "success_case"
INSIGHT = "insight"


class _FallbackTokenizer:
    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return " ".join(tokens)


def _get_tokenizer(model: str = "gpt-4.1"):
    if tiktoken is None:
        return _FallbackTokenizer()
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return _FallbackTokenizer()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _trim_chars(text: str, max_chars: int) -> str:
    clean = normalize_text(text)
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[: max(max_chars - 12, 1)].rstrip() + " [truncated]"


def _jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _is_executor_case(item: dict[str, Any]) -> bool:
    return str(item.get("agent_role", "")).strip() == "executor" or str(item.get("stage", "")).strip() == "EXEC_STEP"


def serialize_executor_trajectory_case(
    case: dict[str, Any],
    *,
    max_chars: int = 2400,
    obs_max_chars: int = 600,
) -> str:
    explicit_memory_text = str(case.get("executor_memory_text") or dict(case.get("metadata") or {}).get("executor_memory_text") or "").strip()
    if explicit_memory_text:
        return explicit_memory_text
    metadata = dict(case.get("metadata") or {})
    trajectory = dict(metadata.get("executor_trajectory") or {})
    task = (
        trajectory.get("task_description")
        or case.get("current_state_text")
        or case.get("state_text")
        or case.get("question_text")
        or ""
    )
    final_output = trajectory.get("final_output") or case.get("final_output") or case.get("target_text") or case.get("plan") or ""
    tool_steps = list(trajectory.get("tool_calls") or [])

    lines: list[str] = [
        "[EXECUTOR_CASE]",
        "[STATE]",
        f"Task: {_trim_chars(str(task), 700) or '[NONE]'}",
        f"Success: {trajectory.get('success_signal', case.get('success_signal', case.get('reward', 'unknown')))}",
        "",
        "[TOOL_TRACE]",
    ]
    if tool_steps:
        for idx, step in enumerate(tool_steps, start=1):
            tool_name = str(step.get("tool_name") or step.get("resolved_name") or step.get("requested_name") or "unknown")
            arguments = _jsonish(step.get("arguments", step.get("arguments_raw", {})))
            observation = step.get("observation_summary") or step.get("observation") or step.get("result_preview") or step.get("error") or ""
            decision = step.get("decision_rationale") or "Use the observation to decide whether to continue, verify, or finish."
            lines.extend(
                [
                    f"Step {idx}:",
                    "[TOOL_CALL]",
                    _trim_chars(f"{tool_name}({arguments})", 500) or "[NONE]",
                    "[OBSERVATION]",
                    _trim_chars(str(observation), obs_max_chars) or "[NONE]",
                    "[DECISION]",
                    _trim_chars(str(decision), 420) or "[NONE]",
                    "",
                ]
            )
    else:
        lines.append("[NONE]")
        lines.extend(
            [
                "",
                "[FORMAT_NOTE]",
                "Legacy executor memory without tool trajectory.",
            ]
        )

    lines.extend(
        [
            "[OUTPUT]",
            _trim_chars(str(final_output), 800) or "[NONE]",
            "",
            "[WHY_USEFUL]",
            "This case is useful for choosing tool calls, adapting search/crawl behavior, and extracting answer evidence.",
        ]
    )
    rendered = "\n".join(lines).strip()
    if max_chars > 0 and len(rendered) > max_chars:
        return rendered[: max(max_chars - 12, 1)].rstrip() + " [truncated]"
    return rendered


def _memory_type(item: dict[str, Any]) -> str:
    metadata = dict(item.get("metadata") or {})
    explicit = str(metadata.get("memory_type") or item.get("memory_type") or "").strip()
    if explicit:
        return explicit
    return SUCCESS_CASE if int(item.get("reward", 0) or 0) == 1 else ""


def build_role_memory_entry(item: dict[str, Any]) -> dict[str, str]:
    memory_type = _memory_type(item)
    if memory_type == INSIGHT:
        return {
            "memory_type": INSIGHT,
            "insight": normalize_text(str(item.get("target_text", item.get("insight", "")))),
        }
    if _is_executor_case(item):
        return {
            "memory_type": SUCCESS_CASE,
            "executor_case": serialize_executor_trajectory_case(item),
        }
    return {
        "memory_type": SUCCESS_CASE,
        "output": normalize_text(str(item.get("target_text", item.get("plan", "")))),
    }


def render_role_memory_entry(entry: dict[str, str]) -> str:
    if entry.get("insight"):
        return f"[INSIGHT] {normalize_text(entry['insight'])}".strip()
    if entry.get("executor_case"):
        return entry["executor_case"].strip()
    chunks: list[str] = []
    for key, label in POSITIVE_OUTPUT_FIELD_ORDER:
        value = normalize_text(entry.get(key, ""))
        if value:
            chunks.append(f"{label} {value}")
    return "\n".join(chunks).strip()


def build_positive_output_entry(item: dict[str, Any]) -> dict[str, str]:
    return build_role_memory_entry(item)


def render_positive_output_entry(entry: dict[str, str]) -> str:
    return render_role_memory_entry(entry)


def _token_len(text: str, model: str = "gpt-4.1") -> int:
    if not text:
        return 0
    enc = _get_tokenizer(model)
    return len(enc.encode(text))


def _trim_to_budget(text: str, budget: int, model: str = "gpt-4.1") -> str:
    enc = _get_tokenizer(model)
    tokens = enc.encode(text)
    if len(tokens) <= budget:
        return text
    return enc.decode(tokens[:budget]).strip()


def render_role_memory(
    ranked_items: list[dict[str, Any]],
    budget_tokens: int,
    model: str = "gpt-4.1",
    bounded_budget_tokens: int | None = None,
) -> dict[str, Any]:
    entries = [build_role_memory_entry(item) for item in ranked_items]
    kept = [entry for entry in entries if entry.get("output") or entry.get("executor_case") or entry.get("insight")]
    dropped = len(entries) - len(kept)

    def render_all(payload: list[dict[str, str]]) -> str:
        blocks = []
        success_idx = 0
        insight_idx = 0
        for entry in payload:
            block = render_role_memory_entry(entry)
            if block:
                if entry.get("memory_type") == INSIGHT or entry.get("insight"):
                    insight_idx += 1
                    label = f"INSIGHT_{insight_idx}"
                else:
                    success_idx += 1
                    label = f"SUCCESS_CASE_{success_idx}"
                blocks.append(f"[{label}]\n{block}")
        return "\n\n".join(blocks).strip()

    rendered = render_all(kept)
    while len(kept) > 1 and _token_len(rendered, model) > budget_tokens:
        kept.pop()
        dropped += 1
        rendered = render_all(kept)

    if kept and _token_len(rendered, model) > budget_tokens:
        tail = kept[-1]
        key = "insight" if tail.get("insight") else "executor_case" if tail.get("executor_case") else "output"
        tail[key] = _trim_to_budget(tail.get(key, ""), budget_tokens, model)
        rendered = render_all(kept)

    text = rendered
    b_text = bounded_budget_tokens if bounded_budget_tokens is not None else min(budget_tokens, max(budget_tokens // 2, 1))
    bounded_text = _trim_to_budget(text, b_text, model)
    return {
        "text": text,
        "bounded_text": bounded_text,
        "text_tokens": _token_len(text, model),
        "bounded_text_tokens": _token_len(bounded_text, model),
        "kept_items": len(kept),
        "dropped_items": dropped,
        "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "bounded_hash": hashlib.sha256(bounded_text.encode("utf-8")).hexdigest(),
    }


def render_positive_output_memory(
    ranked_items: list[dict[str, Any]],
    budget_tokens: int,
    model: str = "gpt-4.1",
    bounded_budget_tokens: int | None = None,
) -> dict[str, Any]:
    return render_role_memory(
        ranked_items,
        budget_tokens=budget_tokens,
        model=model,
        bounded_budget_tokens=bounded_budget_tokens,
    )
