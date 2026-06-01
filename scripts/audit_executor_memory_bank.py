#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

NEED_NEXT_TEXT = "search again, crawl, extract, verify, or finish"
NEED_NEXT_SENTENCE = "Decide whether to search again, crawl, extract, verify, or finish."
NEED_NEXT_BLOCK_RE = re.compile(
    r"\n?\[NEED_NEXT\]\s*(?:\r?\n)?\s*"
    r"(?:Decide whether to search again, crawl, extract, verify, or finish\.)?",
    re.IGNORECASE,
)
HARMFUL_OUTPUT_RE = re.compile(
    r"cannot|unable|no\s+information|no\s+available\s+information|insufficient|无法|不能|不确定",
    re.IGNORECASE,
)
SYNTHESIS_TASK_RE = re.compile(
    r"\b(compare|conclude|summari[sz]e|synthesis|synthesi[sz]e|final|decide)\b|"
    r"\bdetermine\s+(if|whether)\b|"
    r"\bidentify\s+which\b",
    re.IGNORECASE,
)
ERROR_ONLY_RE = re.compile(
    r"^\s*(?:error|exception|traceback|timeout|timed out|failed|failure|tool error|"
    r"permission denied|connection error|network error|rate limit|not found)\b",
    re.IGNORECASE,
)
SECTION_RE_TEMPLATE = r"\[{section}\]\s*(.*?)(?=\n\[[A-Z0-9_]+\]|\Z)"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_executor_row(row: dict[str, Any]) -> bool:
    return str(row.get("stage", "")).strip() == "EXEC_STEP" or str(row.get("agent_role", "")).strip() == "executor"


def is_positive(row: dict[str, Any]) -> bool:
    return int(row.get("reward", 0) or 0) == 1


def executor_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    return dict(dict(row.get("metadata") or {}).get("executor_trajectory") or {})


def attempted_tool_call_count(row: dict[str, Any]) -> int:
    calls = executor_trajectory(row).get("tool_calls") or []
    return len(calls) if isinstance(calls, list) else 0


def tool_call_count(row: dict[str, Any]) -> int:
    return attempted_tool_call_count(row)


def _meaningful_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.upper() in {"[NONE]", "NONE", "NULL", "N/A"}:
        return ""
    return text


def _is_error_only_text(text: str) -> bool:
    value = _meaningful_text(text)
    if not value:
        return False
    return bool(ERROR_ONLY_RE.search(value))


def _is_pure_error_call(call: dict[str, Any]) -> bool:
    error = _meaningful_text(call.get("error"))
    observation = _meaningful_text(call.get("observation"))
    observation_summary = _meaningful_text(call.get("observation_summary"))
    observed_text = "\n".join(part for part in (observation, observation_summary) if part)
    if error and (not observed_text or _is_error_only_text(observed_text)):
        return True
    return bool(observed_text and _is_error_only_text(observed_text))


def _tool_name(call: dict[str, Any]) -> str:
    return _meaningful_text(call.get("tool_name"))


def informative_tool_call_count(row: dict[str, Any]) -> int:
    calls = executor_trajectory(row).get("tool_calls") or []
    if not isinstance(calls, list):
        return 0
    count = 0
    for call in calls:
        if not isinstance(call, dict):
            continue
        observation = _meaningful_text(call.get("observation"))
        observation_summary = _meaningful_text(call.get("observation_summary"))
        if _tool_name(call) and (observation or observation_summary) and not _is_pure_error_call(call):
            count += 1
    return count


def has_real_tool_trajectory(row: dict[str, Any]) -> bool:
    return informative_tool_call_count(row) > 0 and not bool(executor_trajectory(row).get("legacy_format", False))


def row_text(row: dict[str, Any]) -> str:
    metadata = dict(row.get("metadata") or {})
    trajectory = executor_trajectory(row)
    parts = [
        row.get("state_text", ""),
        row.get("current_state_text", ""),
        row.get("target_text", ""),
        metadata.get("executor_memory_text", ""),
        trajectory.get("final_output", ""),
    ]
    return "\n".join(str(part or "") for part in parts)


def has_need_next_pollution(row: dict[str, Any]) -> bool:
    text = row_text(row)
    return "[NEED_NEXT]" in text or NEED_NEXT_TEXT in text


def sanitize_legacy_need_next_text(text: str) -> str:
    sanitized = NEED_NEXT_BLOCK_RE.sub("", text)
    sanitized = sanitized.replace(NEED_NEXT_SENTENCE, "")
    sanitized = sanitized.replace(NEED_NEXT_TEXT, "")
    return re.sub(r"\n{3,}", "\n\n", sanitized).strip()


def sanitize_legacy_need_next(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_legacy_need_next_text(value)
    if isinstance(value, list):
        return [sanitize_legacy_need_next(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_legacy_need_next(item) for key, item in value.items()}
    return value


def output_text(row: dict[str, Any]) -> str:
    trajectory = executor_trajectory(row)
    return "\n".join(
        str(part or "")
        for part in (
            row.get("target_text", ""),
            trajectory.get("final_output", ""),
            dict(row.get("metadata") or {}).get("executor_memory_text", ""),
        )
    )


def has_harmful_cannot_output(row: dict[str, Any]) -> bool:
    return bool(HARMFUL_OUTPUT_RE.search(final_output_text(row)))


def final_output_text(row: dict[str, Any]) -> str:
    trajectory = executor_trajectory(row)
    return "\n".join(
        str(part or "")
        for part in (
            trajectory.get("final_output", ""),
            row.get("target_text", ""),
        )
    )


def is_synthesis_executor_row(row: dict[str, Any]) -> bool:
    metadata = dict(row.get("metadata") or {})
    trajectory = executor_trajectory(row)
    task = str(
        metadata.get("raw_task_description")
        or metadata.get("task_description")
        or trajectory.get("task_description")
        or row.get("current_state_text")
        or row.get("state_text")
        or ""
    )
    return bool(SYNTHESIS_TASK_RE.search(task))


def _section_text(text: str, section: str) -> str:
    match = re.search(SECTION_RE_TEMPLATE.format(section=re.escape(section)), text, flags=re.DOTALL)
    return _meaningful_text(match.group(1)) if match else ""


def _partial_result_text(row: dict[str, Any]) -> str:
    state_texts = [
        str(row.get("current_state_text") or ""),
        str(row.get("state_text") or ""),
    ]
    for text in state_texts:
        partial = _section_text(text, "PARTIAL_RESULT")
        if partial:
            return partial
    return ""


def has_upstream_evidence(row: dict[str, Any]) -> bool:
    trajectory = executor_trajectory(row)
    observations = trajectory.get("observations") or []
    if isinstance(observations, list) and any(_meaningful_text(item) for item in observations):
        return True
    if _partial_result_text(row):
        return True
    for field in ("subtask_memory_text", "tool_memory_text"):
        if _meaningful_text(row.get(field)):
            return True
    return False


def classify_executor_memory(row: dict[str, Any]) -> tuple[str, list[str]]:
    if not is_executor_row(row):
        return "planner", []
    reasons: list[str] = []
    if has_need_next_pollution(row):
        reasons.append("need_next_pollution")
    attempted = attempted_tool_call_count(row)
    informative = informative_tool_call_count(row)
    cannot = has_harmful_cannot_output(row)
    if attempted > 0 and cannot:
        reasons.append("tool_call_cannot_output")
        return "marked_tool_cannot", reasons
    if attempted == 0 and cannot:
        reasons.append("no_tool_cannot_output")
        return "harmful", reasons
    if informative > 0:
        return "action_oriented", reasons
    if attempted > 0:
        reasons.append("attempted_tool_without_informative_observation")
        return "weak_action_oriented", reasons
    if is_synthesis_executor_row(row):
        if has_upstream_evidence(row):
            return "synthesis", reasons
        reasons.append("synthesis_without_upstream_evidence")
        return "synthesis_candidate", reasons
    reasons.append("legacy_no_tool_non_synthesis")
    return "synthesis_candidate", reasons


def annotate_executor_row(row: dict[str, Any], memory_type: str, reasons: list[str]) -> dict[str, Any]:
    updated = json.loads(json.dumps(row, ensure_ascii=False))
    if is_executor_row(updated):
        metadata = dict(updated.get("metadata") or {})
        metadata["executor_memory_type"] = memory_type
        metadata["filter_reason"] = reasons
        updated["metadata"] = metadata
    return updated


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    executor_rows = [row for row in rows if is_executor_row(row)]
    positive_executor_rows = [row for row in executor_rows if is_positive(row)]
    type_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in executor_rows:
        memory_type, reasons = classify_executor_memory(row)
        type_counts[memory_type] += 1
        reason_counts.update(reasons)
    real_positive = [row for row in positive_executor_rows if has_real_tool_trajectory(row)]
    harmful_no_tool = [
        row
        for row in executor_rows
        if attempted_tool_call_count(row) == 0 and has_harmful_cannot_output(row)
    ]
    positive_harmful_no_tool = [row for row in harmful_no_tool if is_positive(row)]
    attempted_tool_rows = [row for row in executor_rows if attempted_tool_call_count(row) > 0]
    informative_tool_rows = [row for row in executor_rows if informative_tool_call_count(row) > 0]
    return {
        "rows_total": len(rows),
        "planner_rows": len(rows) - len(executor_rows),
        "executor_rows": len(executor_rows),
        "positive_executor_rows": len(positive_executor_rows),
        "need_next_rows": sum(1 for row in executor_rows if has_need_next_pollution(row)),
        "tool_calls_zero_executor_rows": sum(1 for row in executor_rows if attempted_tool_call_count(row) == 0),
        "attempted_tool_call_executor_rows": len(attempted_tool_rows),
        "informative_tool_call_executor_rows": len(informative_tool_rows),
        "weak_action_oriented_rows": sum(
            1 for row in executor_rows if classify_executor_memory(row)[0] == "weak_action_oriented"
        ),
        "marked_tool_cannot_rows": sum(
            1 for row in executor_rows if classify_executor_memory(row)[0] == "marked_tool_cannot"
        ),
        "synthesis_candidate_rows": sum(
            1 for row in executor_rows if classify_executor_memory(row)[0] == "synthesis_candidate"
        ),
        "no_tool_cannot_rows": len(harmful_no_tool),
        "positive_no_tool_cannot_rows": len(positive_harmful_no_tool),
        "positive_executor_real_tool_trajectory_rows": len(real_positive),
        "positive_executor_real_tool_trajectory_ratio": (
            len(real_positive) / len(positive_executor_rows) if positive_executor_rows else 0.0
        ),
        "executor_memory_type_counts": dict(type_counts),
        "filter_reason_counts": dict(reason_counts),
    }


def audit_files(paths: list[str | Path]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for path in paths:
        rows = load_jsonl(path)
        files[str(path)] = audit_rows(rows)
        aggregate_rows.extend(rows)
    return {
        "files": files,
        "aggregate": audit_rows(aggregate_rows),
    }


def render_audit_markdown(report: dict[str, Any]) -> str:
    lines = ["# Executor Memory Bank Audit", "", "## Aggregate", ""]
    aggregate = report.get("aggregate", {})
    for key, value in aggregate.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Files", ""])
    for path, stats in dict(report.get("files") or {}).items():
        lines.append(f"### `{path}`")
        for key, value in stats.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit StageWeaver executor memory bank quality.")
    parser.add_argument("bank_jsonl", nargs="+", help="Stage bank JSONL file(s) to audit.")
    parser.add_argument("--output_json", default="", help="Optional audit report JSON path.")
    parser.add_argument("--output_md", default="", help="Optional audit report Markdown path.")
    args = parser.parse_args()

    report = audit_files(args.bank_jsonl)
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_audit_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
