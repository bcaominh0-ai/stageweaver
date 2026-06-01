#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.audit_executor_memory_bank import (
        annotate_executor_row,
        audit_files,
        classify_executor_memory,
        is_executor_row,
        is_positive,
        load_jsonl,
        render_audit_markdown,
        sanitize_legacy_need_next,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from audit_executor_memory_bank import (
        annotate_executor_row,
        audit_files,
        classify_executor_memory,
        is_executor_row,
        is_positive,
        load_jsonl,
        render_audit_markdown,
        sanitize_legacy_need_next,
        write_jsonl,
    )


HARD_FILTER_REASONS = {
    "positive_no_tool_cannot_output",
    "positive_legacy_no_tool_non_synthesis",
}


def should_filter_executor_row(row: dict[str, Any], reasons: list[str]) -> bool:
    return is_positive(row) and any(reason in HARD_FILTER_REASONS for reason in reasons)


def filter_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not is_executor_row(row):
            kept.append(sanitize_legacy_need_next(row))
            continue
        memory_type, reasons = classify_executor_memory(row)
        annotated = annotate_executor_row(row, memory_type, reasons)
        if should_filter_executor_row(row, reasons):
            annotated = annotate_executor_row(row, "filtered", reasons)
            filtered.append(annotated)
            continue
        kept.append(sanitize_legacy_need_next(annotated))
    return kept, filtered


def filter_bank_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    kept, filtered = filter_rows(rows)
    write_jsonl(output_path, kept)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(kept),
        "filtered_executor_rows": len(filtered),
        "filtered_positive_executor_rows": sum(1 for row in filtered if is_positive(row)),
        "sanitized_executor_rows": sum(1 for row in kept if is_executor_row(row)),
        "filtered_source_ids": [str(row.get("source_id", "")) for row in filtered],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter harmful StageWeaver executor positive memory rows.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument(
        "--output_dir",
        default="result/stageweaver/current/stage_bank_filtered",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "stage_bank_train.filtered.jsonl"
    val_out = output_dir / "stage_bank_val.filtered.jsonl"

    filter_report = {
        "train": filter_bank_file(args.train_jsonl, train_out),
        "val": filter_bank_file(args.val_jsonl, val_out),
    }
    input_audit_report = audit_files([args.train_jsonl, args.val_jsonl])
    filtered_audit_report = audit_files([train_out, val_out])
    report = {
        "filter": filter_report,
        "input_audit": input_audit_report,
        "filtered_audit": filtered_audit_report,
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "audit_report.md").write_text(
        "# Filtered Executor Memory Bank Report\n\n"
        "## Filter Summary\n\n"
        f"```json\n{json.dumps(filter_report, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Input Audit\n\n"
        + render_audit_markdown(input_audit_report)
        + "\n## Filtered Audit\n\n"
        + render_audit_markdown(filtered_audit_report),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
