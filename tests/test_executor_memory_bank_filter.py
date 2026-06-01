from __future__ import annotations

import unittest

from scripts.audit_executor_memory_bank import audit_rows, classify_executor_memory
from scripts.filter_executor_memory_bank import filter_rows


def _planner_row() -> dict:
    return {
        "stage": "PLAN_INIT",
        "agent_role": "planner",
        "reward": 1,
        "source_id": "planner-1",
        "state_text": "question",
        "target_text": '{"plan": []}',
        "metadata": {},
    }


def _executor_row(
    *,
    source_id: str = "executor-1",
    reward: int = 1,
    task: str = "Find a factual answer.",
    target: str = "[RETURN] unable to find information",
    tool_calls: list[dict] | None = None,
    state_suffix: str = "",
) -> dict:
    calls = list(tool_calls or [])
    return {
        "stage": "EXEC_STEP",
        "agent_role": "executor",
        "reward": reward,
        "source_id": source_id,
        "state_text": f"[EXECUTOR_TASK]\n{task}{state_suffix}",
        "current_state_text": f"[EXECUTOR_TASK]\n{task}{state_suffix}",
        "target_text": target,
        "metadata": {
            "raw_task_description": task,
            "executor_trajectory": {
                "task_description": task,
                "tool_calls": calls,
                "legacy_format": not bool(calls),
                "final_output": target,
            },
            "executor_memory_text": target,
        },
    }


class ExecutorMemoryBankFilterTests(unittest.TestCase):
    def test_audit_detects_need_next_pollution(self) -> None:
        row = _executor_row(
            target="[RETURN] answer",
            tool_calls=[{"tool": "search", "arguments": {"query": "Ada Lovelace"}}],
            state_suffix="\n[NEED_NEXT]\nDecide whether to search again, crawl, extract, verify, or finish.",
        )
        stats = audit_rows([row])

        self.assertEqual(stats["executor_rows"], 1)
        self.assertEqual(stats["need_next_rows"], 1)
        memory_type, reasons = classify_executor_memory(row)
        self.assertEqual(memory_type, "action_oriented")
        self.assertIn("need_next_pollution", reasons)

    def test_need_next_pollution_is_sanitized_not_dropped(self) -> None:
        row = _executor_row(
            source_id="executor-need-next",
            target="[RETURN] answer",
            tool_calls=[{"tool": "search", "arguments": {"query": "Ada Lovelace"}}],
            state_suffix="\n[NEED_NEXT]\nDecide whether to search again, crawl, extract, verify, or finish.",
        )
        row["source_text"] = "[CURRENT_STATE]\n[NEED_NEXT]\nDecide whether to search again, crawl, extract, verify, or finish."
        row["metadata"]["executor_memory_text"] = (
            "[EXECUTOR_CASE]\n[STATE]\n[NEED_NEXT]\n"
            "Decide whether to search again, crawl, extract, verify, or finish.\n[OUTPUT]\nanswer"
        )

        kept, filtered = filter_rows([row])

        self.assertEqual(filtered, [])
        self.assertEqual(len(kept), 1)
        sanitized = kept[0]
        for field in ("state_text", "current_state_text", "source_text"):
            self.assertNotIn("[NEED_NEXT]", sanitized[field])
            self.assertNotIn("search again, crawl, extract, verify, or finish", sanitized[field])
        self.assertNotIn("[NEED_NEXT]", sanitized["metadata"]["executor_memory_text"])
        self.assertNotIn(
            "search again, crawl, extract, verify, or finish",
            sanitized["metadata"]["executor_memory_text"],
        )
        self.assertIn("need_next_pollution", sanitized["metadata"]["filter_reason"])

    def test_positive_no_tool_cannot_output_is_filtered(self) -> None:
        row = _executor_row(
            reward=1,
            task="Find the nationality of Ada Lovelace.",
            target="[RETURN] I cannot find enough information.",
            tool_calls=[],
        )
        kept, filtered = filter_rows([row])

        self.assertEqual(kept, [])
        self.assertEqual(len(filtered), 1)
        self.assertIn("positive_no_tool_cannot_output", filtered[0]["metadata"]["filter_reason"])

    def test_planner_rows_are_never_filtered(self) -> None:
        planner = _planner_row()
        harmful_executor = _executor_row(
            reward=1,
            target="[RETURN] unable to answer",
            tool_calls=[],
        )
        kept, filtered = filter_rows([planner, harmful_executor])

        self.assertEqual([row["source_id"] for row in kept], ["planner-1"])
        self.assertEqual(len(filtered), 1)


if __name__ == "__main__":
    unittest.main()
