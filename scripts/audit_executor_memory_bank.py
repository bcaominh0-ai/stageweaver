#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

NEED_NEXT_TEXT = "search again, crawl, extract, verify, or finish"
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


def tool_call_count(row: dict[str, Any]) -> int:
    calls = executor_trajectory(row).get("tool_calls") or []
    return len(calls) if isinstance(calls, list) else 0


def has_real_tool_trajectory(row: dict[str, Any]) -> bool:
    return tool_call_count(row) > 0 and not bool(executor_trajectory(row).get("legacy_format", False))


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
    return bool(HARMFUL_OUTPUT_RE.search(output_text(row)))


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


def classify_executor_memory(row: dict[str, Any]) -> tuple[str, list[str]]:
    if not is_executor_row(row):
        return "planner", []
    reasons: list[str] = []
    if has_need_next_pollution(row):
        reasons.append("need_next_pollution")
    if is_positive(row) and tool_call_count(row) == 0 and has_harmful_cannot_output(row):
        reasons.append("positive_no_tool_cannot_output")
    if is_positive(row) and tool_call_count(row) == 0 and not is_synthesis_executor_row(row):
        reasons.append("positive_legacy_no_tool_non_synthesis")
    if reasons:
        return "filtered", reasons
    if is_synthesis_executor_row(row) and tool_call_count(row) == 0:
        return "synthesis", []
    return "action_oriented", []


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
        for row in positive_executor_rows
        if tool_call_count(row) == 0 and has_harmful_cannot_output(row)
    ]
    return {
        "rows_total": len(rows),
        "planner_rows": len(rows) - len(executor_rows),
        "executor_rows": len(executor_rows),
        "positive_executor_rows": len(positive_executor_rows),
        "need_next_rows": sum(1 for row in executor_rows if has_need_next_pollution(row)),
        "tool_calls_zero_executor_rows": sum(1 for row in executor_rows if tool_call_count(row) == 0),
        "positive_no_tool_cannot_rows": len(harmful_no_tool),
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
