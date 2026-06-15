from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEGACY_MEMENTO_FILENAMES = {"memory.jsonl", "memento_text_seed_memory.jsonl"}


def _build_stage_bank_hint(path: Path) -> str:
    return (
        f"Missing current-protocol Memento-Text planner stage bank: {path}. "
        "Run build_stageweaver_bank first to create result/stageweaver/current/stage_bank/stage_bank_train.jsonl. "
        "There is no fallback to legacy memory.jsonl or memento_text_seed_memory.jsonl."
    )


def validate_memento_stage_bank_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.name in LEGACY_MEMENTO_FILENAMES:
        raise FileNotFoundError(
            "Current Memento-Text no longer accepts legacy seed-memory artifacts. "
            f"Received {resolved.name!r}. Run build_stageweaver_bank first to create "
            "result/stageweaver/current/stage_bank/stage_bank_train.jsonl."
        )
    if not resolved.is_file():
        raise FileNotFoundError(_build_stage_bank_hint(resolved))
    return resolved


def load_memento_planner_cases(path: str | Path) -> list[dict[str, Any]]:
    resolved = validate_memento_stage_bank_path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stage = str(obj.get("stage", "PLAN_INIT")).strip()
            agent_role = str(obj.get("agent_role", "")).strip()
            if agent_role == "executor" or stage == "EXEC_STEP":
                continue
            case = str(obj.get("state_text", obj.get("question_text", obj.get("question", "")))).strip()
            plan = str(obj.get("target_text", obj.get("plan", ""))).strip()
            if not case or not plan:
                continue
            metadata = dict(obj.get("metadata") or {})
            memory_type = str(metadata.get("memory_type") or obj.get("memory_type") or "").strip()
            reward = int(
                obj.get(
                    "reward",
                    1 if memory_type == "success_case" or str(obj.get("case_label", "")).lower() == "positive" else 0,
                )
            )
            rows.append(
                {
                    "state_text": case,
                    "question": str(obj.get("question_text", obj.get("question", case))).strip(),
                    "target_text": plan,
                    "source_id": str(obj.get("source_id", f"memory-{idx:05d}")),
                    "stage": stage or "PLAN_INIT",
                    "reward": reward,
                    "agent_role": agent_role or "planner",
                    "current_state_text": str(obj.get("current_state_text", case)).strip(),
                    "metadata": metadata or {"case_label": obj.get("case_label", "")},
                }
            )
    if not rows:
        raise RuntimeError(
            f"No planner tuples were found in the current-protocol Memento-Text stage bank: {resolved}. "
            "Run build_stageweaver_bank first and make sure stage_bank_train.jsonl contains planner tuples."
        )
    return rows
